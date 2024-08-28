# Copyright 2024 Alibaba Group Holding Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""old policy inference"""

import inspect
import numpy as np
import torch
import torch.nn.functional as F

from megatron.core import mpu
from megatron.training import arguments
from megatron.training import get_args, get_tokenizer
from megatron.training import print_rank_0
from megatron.training.global_vars import get_tensorboard_writer
from megatron.inference.text_generation.communication import broadcast_float_list, \
    broadcast_int_list, broadcast_tensor
from megatron.inference.text_generation.generation import generate_tokens_probs_and_return_on_first_stage
from megatron.inference.text_generation.forward_step import ForwardStep
from megatron.training import get_model

from chatlearn import MegatronModule
from chatlearn.utils import to_device
from chatlearn.utils.megatron_utils import load_checkpoint
from examples.megatron.data.prompt_dataset import PromptPipeline
from .policy_model import PolicyModel
from .utils import tensorboard_scalar_dict, get_loss_mask, get_eos_id


class PolicyInference(MegatronModule):
    """Policy Megatron Inference"""

    def add_extra_args(self, parser):
        group = parser.add_argument_group(title='text generation')

        group.add_argument("--temperature", type=float, default=1.0,
                           help='Sampling temperature.')
        group.add_argument("--top_p", type=float, default=0.0,
                           help='Top p sampling.')
        group.add_argument("--top_k", type=int, default=0,
                           help='Top k sampling.')
        if '--use-attn-acc' not in inspect.getsource(arguments._add_training_args):
            group.add_argument('--use-attn-acc', action='store_true',
                            help='use attention-acc kernel')
        return parser

    def setup(self):
        self.args = get_args()
        # Set up model and load checkpoint
        model = get_model(self.model_provider, wrap_with_ddp=False)
        self.tokenizer = get_tokenizer()
        if self.args.load and self.src_parameter_model is None:
            # if self.src_parameter_model is not None, we broadcast parameters from src_parameter_model to current model
            torch.distributed.barrier()
            load_checkpoint(model, None, None, adaptive_parallel_strategy=self.args.adaptive_parallel_strategy_on_checkpoint)
            torch.distributed.barrier()
        assert len(model) == 1, "Above condition should have caught this"
        self.model = model[0]
        self.model.eval()

        if hasattr(self.args,
                   "use_eod_token_for_early_termination") and not self.args.use_eod_token_for_early_termination:
            use_eod_token_for_early_termination = False
            print_rank_0(
                f"use_eod_token_for_early_termination: {use_eod_token_for_early_termination} for benchmark only, " \
                + "please set it to True for real application")

        # this is sum
        get_args().entropy_sum = 0

        # init num
        get_args().entropy_num = 0
        get_args().latest_entropies = []

    def build_dataset(self, train_prompts, is_eval=False):
        args = get_args()
        max_prompt_length = (
            args.seq_length - args.max_new_tokens
        )

        # TODO: read from files
        prompts_dataset = PromptPipeline(
            train_prompts, max_prompt_length, get_tokenizer()
        )
        return prompts_dataset

    def model_provider(self, pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0('building GPT model ...')
        model = PolicyModel(num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process)

        return model

    def decode(self,
               no_padded_query_ids,
               all_tokens,
               ):
        """
        Decode tensor generations into lists of strings (`samples`: List[str], `prompts`: List[str], `outputs`: List[str])
        """

        tokenizer = get_tokenizer().tokenizer
        # Assuming prompts were left-padded
        prompt_sizes = [len(q) for q in no_padded_query_ids]

        str_samples, str_prompts, str_outputs, response_ids = [], [], [], []

        kwargs = {}
        if 'skip_special_tokens' in inspect.getfullargspec(tokenizer.decode).args:
            kwargs = {"skip_special_tokens": True}
        for prompt, sample, prompt_size in zip(no_padded_query_ids, all_tokens, prompt_sizes):
            output_start_ix = prompt_size
            str_prompt = tokenizer.decode(
                prompt.tolist(), **kwargs
            )
            str_output = tokenizer.decode(
                sample[output_start_ix:].tolist(), **kwargs
            )
            response_id = sample[output_start_ix:]
            response_ids.append(response_id)

            str_prompts.append(str_prompt)
            str_outputs.append(str_output)

            sample = str_prompt + str_output

            str_samples.append(sample)

        return str_samples, str_prompts, str_outputs, response_ids

    def _tokenize_prompts_and_batch(self, prompts_tokens, tokens_to_generate):
        """Given a set of prompts and number of tokens to generate:
        prompts_tokens:
            - tokenize prompts
            - set the sequence length to be the max of length of prompts
              plus the number of tokens we would like to generate
            - pad all the sequences to this length so we can convert them
              into a 2D tensor.
        """

        # Tokenize all the prompts.
        tokenizer = get_tokenizer()

        # Now we have a list of list of tokens which each list has a different
        # size. We want to extend this list to:
        #   - incorporate the tokens that need to be generated
        #   - make all the sequences equal length.
        # Get the prompts length.
        prompts_length = [len(prompt_token)
                          for prompt_token in prompts_tokens]
        # Get the max prompts length.
        max_prompt_len = max(prompts_length)

        # Number of tokens in the each sample of the batch.
        samples_length = max_prompt_len + tokens_to_generate
        # Now update the list of list to be of the same size: samples_length.

        prompts_tokens = [
            F.pad(
                prompts_token,
                (0, samples_length - len(prompts_token)),
                value=tokenizer.eod,  # just pad_token_id
            )
            for prompts_token in prompts_tokens
        ]
        prompts_tokens_tensor = torch.vstack(prompts_tokens).to(torch.cuda.current_device())
        assert prompts_tokens_tensor.size(1) == samples_length, "pad to the query_size + max generate size"

        # Now we are in a structured format, we can convert to tensors.

        # print(f"after pad prompts_tokens_tensor size {prompts_tokens_tensor.size()}")
        prompts_length_tensor = torch.tensor(prompts_length, dtype=torch.long, device='cuda')
        # assert torch.all(prompts_length_tensor ==  max_prompt_len), "because left padded"
        return prompts_tokens_tensor, prompts_length_tensor

    def tokenize_prompts(self, prompts_ids=None, tokens_to_generate=None, rank=0):
        """Tokenize prompts and make them avaiable on all ranks."""

        # On all ranks set to None so we can pass them to functions
        sizes_list = None
        prompts_tokens_cuda_long_tensor = None
        prompts_length_cuda_long_tensor = None

        # On the specified rank, build the above.
        if torch.distributed.get_rank() == rank:
            assert prompts_ids is not None
            assert tokens_to_generate is not None
            # Tensor of tokens padded and their unpadded length.
            prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor = \
                self._tokenize_prompts_and_batch(prompts_ids, tokens_to_generate)
            # We need the sizes of these tensors for the boradcast
            sizes_list = [prompts_tokens_cuda_long_tensor.size(0),  # Batch size
                          prompts_tokens_cuda_long_tensor.size(1)]  # Sequence length

        # First, broadcast the sizes.
        sizes_tensor = broadcast_int_list(2, int_list=sizes_list, rank=rank)

        # Now that we have the sizes, we can boradcast the tokens
        # and length tensors.
        sizes = sizes_tensor.tolist()
        prompts_tokens_cuda_long_tensor = broadcast_tensor(
            sizes, torch.int64, tensor=prompts_tokens_cuda_long_tensor, rank=rank)
        prompts_length_cuda_long_tensor = broadcast_tensor(
            sizes[0], torch.int64, tensor=prompts_length_cuda_long_tensor,
            rank=rank)

        return prompts_tokens_cuda_long_tensor, prompts_length_cuda_long_tensor

    def generate(self, model,
                 prompts_ids=None,
                 tokens_to_generate=0,
                 return_output_log_probs=False,
                 top_k_sampling=0,
                 top_p_sampling=0.0,
                 temperature=1.0,
                 add_BOS=False,
                 use_eod_token_for_early_termination=True,
                 stop_on_double_eol=False,
                 stop_on_eol=False,
                 random_seed=-1):
        """Given prompts and input parameters, run inference and return:
           tokens: prompts plus the generated tokens.
           lengths: length of the prompt + generations. Note that we can
               discard tokens in the tokens tensor that are after the
               corresponding length.
           output_log_probs: log probs of the tokens.
        """

        # Make sure input params are avaialble to all ranks.
        values = [tokens_to_generate,
                  return_output_log_probs,
                  top_k_sampling, top_p_sampling,
                  temperature, add_BOS, use_eod_token_for_early_termination,
                  stop_on_double_eol,
                  stop_on_eol,
                  random_seed]
        values_float_tensor = broadcast_float_list(10, float_list=values)
        # tokens_to_generate = int(values_float_tensor[0].item())
        return_output_log_probs = bool(values_float_tensor[1].item())
        top_k_sampling = int(values_float_tensor[2].item())
        top_p_sampling = values_float_tensor[3].item()
        temperature = values_float_tensor[4].item()
        # add_BOS = bool(values_float_tensor[5].item())
        use_eod_token_for_early_termination = bool(values_float_tensor[6].item())
        stop_on_double_eol = bool(values_float_tensor[7].item())
        stop_on_eol = bool(values_float_tensor[8].item())
        random_seed = int(values_float_tensor[9].item())

        if random_seed != -1:
            torch.random.manual_seed(random_seed)

        # Tokenize prompts and get the batch.
        # Note that these tensors are broadcaseted to all ranks.
        if torch.distributed.get_rank() == 0:
            assert prompts_ids is not None

        prompts_ids, context_length_tensor = self.tokenize_prompts(
            prompts_ids=prompts_ids, tokens_to_generate=tokens_to_generate)

        # Main inference function.
        # Note that the outputs are available on the first stage.
        res = generate_tokens_probs_and_return_on_first_stage(
            model, ForwardStep, prompts_ids, context_length_tensor,
            return_output_log_probs=return_output_log_probs,
            top_k=top_k_sampling,
            top_p=top_p_sampling,
            temperature=temperature,
            use_eod_token_for_early_termination=use_eod_token_for_early_termination,
            stop_on_double_eol=stop_on_double_eol,
            stop_on_eol=stop_on_eol)
        # tokens, generated_sequence_lengths, output_log_probs, None
        return res[0], res[2]

    def replace_all_after_first_stop_sequences_by_pad(self, tokens, all_log_probs, stop_token, prompt_sizes):
        '''
        only replace after the stop tokens in the response ignore in the prompt
        :param tokens:
        :param all_log_probs:
        :param stop_token:
        :param prompt_sizes:
        :return:
        '''
        assert len(tokens.size()) == 2
        assert len(all_log_probs.size()) == 2
        occurrences = (tokens == stop_token).float()

        row_indices = torch.arange(tokens.size(1)).unsqueeze(0).expand(tokens.size(0), -1).to(tokens.device)
        response_mask = (row_indices >= prompt_sizes.unsqueeze(1)).int()

        # mask out the stop appear in the prompt:
        occurrences = occurrences * response_mask

        first_stop_sequence_indices = torch.argmax(occurrences, dim=1)

        # if not found a stop sequence, occurrence will be sum 0 dim1
        not_found_mask = torch.sum(occurrences, dim=1) == 0
        # for the not found one. take stop_sequence = tokens.size(1)-1 before everything else replace tokens afterwards
        first_stop_sequence_indices[not_found_mask] = tokens.size(1) - 1

        eos_id = get_eos_id(self.tokenizer)

        for i in range(tokens.size(0)):
            if first_stop_sequence_indices[i] < tokens.size(1) - 1:
                # if not the last tokne to stop
                tokens[i, first_stop_sequence_indices[i] + 1:] = eos_id

        # because all_log_probs is the logprobs of the tokens[:, 1:], thus index 4 at tokens = index 3 at all_log_prob
        all_log_probs_indices = first_stop_sequence_indices - 1

        for i in range(all_log_probs.size(0)):
            if all_log_probs_indices[i] < all_log_probs.size(1) - 1:
                # if not the last log prob to stop
                all_log_probs[i, all_log_probs_indices[i] + 1:] = 0.0

        return tokens, all_log_probs

    def eval_forward(self, data):
        return self._forward_step(data, 0, eval_mode=True)

    def _forward_step(self, data, iteration, eval_mode: bool):
        '''
        ChatLearn calling
        chatlearn framework source:     policy_output = self.policy.forward_step(query)
        :param data: entire global batch?? micro_batch?
        :return:
            data using current microbatch
            {"all_tokens": tokens,  "str_samples": str_samples,
                "str_prompts": str_prompts, "str_outputs": str_outputs, "logprobs": all_log_probs,
                "no_padded_query_ids": no_padded_query_ids}
        '''

        no_padded_query_ids = to_device('cuda', data["input_ids"])
        # tokens: [b, qs + rs],
        tokens, all_log_probs = self.generate(
            self.model,
            prompts_ids=no_padded_query_ids,
            tokens_to_generate=self.args.max_new_tokens,
            return_output_log_probs=True,
            top_k_sampling=self.args.top_k if not eval_mode else self.args.eval_top_k,
            top_p_sampling=self.args.top_p if not eval_mode else self.args.eval_top_p,
            temperature=self.args.temperature if not eval_mode else self.args.eval_temperature,
            add_BOS=False,
            use_eod_token_for_early_termination=True,
            stop_on_double_eol=False,
            stop_on_eol=False)
        if not mpu.is_pipeline_last_stage():
            # only last pipeline stage has valid data
            return

        _all_tokens_max_len = tokens.size(1)
        assert not torch.isnan(all_log_probs).any(), f"just out old_logprobs {all_log_probs}"

        assert all_log_probs.size(1) == tokens.size(1) - 1, "because first token hsa no log prob logprob[:, 1:]"

        # everything after stop_token in tokens will be pad_token,
        # everything after stop_token_indx -1 in all_log_probs will be 0.0 (its pad since it's used to minus other logprob)
        prompt_sizes = torch.tensor([len(q) for q in no_padded_query_ids], device=tokens.device)
        tokens, all_log_probs = self.replace_all_after_first_stop_sequences_by_pad(tokens, all_log_probs,
                                                                                   stop_token=get_eos_id(self.tokenizer),
                                                                                   prompt_sizes=prompt_sizes)
        assert all_log_probs.size(1) == tokens.size(1) - 1, "because first token hsa no log prob logprob[:, 1:]"

        str_samples, str_prompts, str_outputs, _ = self.decode(
            no_padded_query_ids, tokens
        )

        if not eval_mode and self.args.log_entropy:
            self.log_entropy(iteration)

        assert tokens.size(1) == _all_tokens_max_len, f"tokens size: {tokens.size(1)} " \
                                                      f"_all_tokens_max_len: {_all_tokens_max_len}"

        loss_mask = get_loss_mask(tokens, get_eos_id(get_tokenizer()), prompt_sizes)

        return {"all_tokens": tokens, "str_samples": str_samples,
                "str_prompts": str_prompts, "str_outputs": str_outputs, "logprobs": all_log_probs,
                "no_padded_query_ids": no_padded_query_ids, "loss_mask": loss_mask}

    def forward_step(self, data, iteration=0):
        return self._forward_step(data, iteration, eval_mode=False)

    def log_entropy(self, iteration_for_log):

        # log

        writer = get_tensorboard_writer()
        # RL related stats: global
        if torch.distributed.is_initialized():
            if torch.distributed.get_rank() == (
                torch.distributed.get_world_size() - 1):
                # actual log

                stats_args = {"entropy_mean": get_args().entropy_sum / get_args().entropy_num,
                              "entropy_percentile_20": np.percentile(get_args().latest_entropies, 20),
                              "entropy_percentile_50": np.percentile(get_args().latest_entropies, 50),
                              "entropy_percentile_70": np.percentile(get_args().latest_entropies, 70),
                              "entropy_percentile_90": np.percentile(get_args().latest_entropies, 90),
                              "max_latest_entropy": np.max(get_args().latest_entropies),
                              "mean_latest_entropy": np.mean(get_args().latest_entropies),
                              }
                tensorboard_scalar_dict(writer, prefix=f"policy_inference/replica_id{self.replica_id}",
                                        global_step=iteration_for_log, scalar_dict=stats_args)

        else:
            # actual log
            stats_args = {"entropy_mean": get_args().entropy_sum / get_args().entropy_num,
                          "entropy_percentile_20": np.percentile(get_args().latest_entropies, 20),
                          "entropy_percentile_50": np.percentile(get_args().latest_entropies, 50),
                          "entropy_percentile_70": np.percentile(get_args().latest_entropies, 70),
                          "entropy_percentile_90": np.percentile(get_args().latest_entropies, 90),
                          "max_latest_entropy": np.max(get_args().latest_entropies),
                          "mean_latest_entropy": np.mean(get_args().latest_entropies),
                          }
            tensorboard_scalar_dict(writer, prefix=f"policy_inference/replica_id{self.replica_id}",
                                    global_step=iteration_for_log, scalar_dict=stats_args)

        get_args().entropy_sum = 0
        get_args().entropy_num = 0
