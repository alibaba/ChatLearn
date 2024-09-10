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
"""reference"""

import torch
import torch.nn.functional as F
from megatron.training import get_args, get_tokenizer
from megatron.training import print_rank_0
from megatron.core import mpu
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.training import get_model

from chatlearn.utils import to_device
from chatlearn.utils.megatron_utils import load_checkpoint
from examples.megatron.data.prompt_dataset import DPOPromptPipeline
from .policy_model import PolicyModel
from .mcore_policy_model import MCorePolicyModel
from .utils import get_eos_id, get_padding_length, pad_to_length
from .constants import get_ltor_masks_and_position_ids_rlhf
from .constants import TrainerEngine
from .forward_step import forward_step_helper
from .old_policy_inference import PolicyInference


class PolicyReference(PolicyInference):
    """PolicyReference"""

    def model_provider(self, pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0('building GPT model ...')
        args = get_args()

        if hasattr(args, "parallel_output") and args.parallel_output:
            self._parallel_output = True
            print_rank_0("enable parallel output")
        else:
            self._parallel_output = False
        if args.use_legacy_models:
            model = PolicyModel(num_tokentypes=0, parallel_output=self._parallel_output, pre_process=pre_process,
                                post_process=post_process)
        else:
            model = MCorePolicyModel(parallel_output=self._parallel_output, pre_process=pre_process,
                                     post_process=post_process)

        return model

    def setup(self):
        self.args = get_args()
        # Set up model and load checkpoint
        model = get_model(self.model_provider, wrap_with_ddp=False)
        self.tokenizer = get_tokenizer()
        if self.args.load:
            torch.distributed.barrier()
            load_checkpoint(model, None, None, adaptive_parallel_strategy=self.args.adaptive_parallel_strategy_on_checkpoint)
            torch.distributed.barrier()
        assert len(model) == 1, "Above condition should have caught this"
        self.model = model[0]
        self.model.eval()

        # this is sum
        get_args().entropy_sum = 0

        # init num
        get_args().entropy_num = 0
        get_args().latest_entropies = []

    def build_dataset(self, train_prompts, is_eval=False):
        args = get_args()
        if args.trainer_engine == TrainerEngine.DPO:
            # TODO: read from files
            return DPOPromptPipeline(
                train_prompts, args.seq_length, get_tokenizer()
            )
        return super().build_dataset(train_prompts, is_eval)

    def score_reference_dpo(self, tokens, prompt_id_lens=None, orig_mask=None):
        if get_args().trainer_engine == TrainerEngine.DPO:
            dpo_labels = tokens[:, :]
            tokens_ = tokens[:,:]
        else:
            dpo_labels = tokens[:, 1:]
            tokens_ = tokens[:,:-1]
        attention_mask, position_ids = get_ltor_masks_and_position_ids_rlhf(tokens_)

        ref_nll = forward_step_helper(
            self.model, tokens_, position_ids, attention_mask, pooling=False,
            inference_config={"DPO_labels":dpo_labels, "prompt_id_lens": prompt_id_lens, "orig_mask": orig_mask})
        return ref_nll

    def concatenated_inputs(self, chosen_ids, c_mask, reject_ids, r_mask):
        """Concatenate the chosen and rejected inputs into a single tensor.

        Args:
            batch: A batch of data. Must contain the keys 'chosen_input_ids' and 'rejected_input_ids',
                which are tensors of shape (batch_size, sequence_length).

        Returns:
            A dictionary containing the concatenated inputs under the key 'concatenated_input_ids'.
        """
        max_length = max(chosen_ids.shape[1], reject_ids.shape[1])
        pad_value = get_eos_id(get_tokenizer())
        tp_size = self.tensor_model_parallel_size()
        sp_enabled = self.megatron_args.sequence_parallel

        max_length = get_padding_length(sp_enabled, tp_size, max_length)

        inputs_ids = torch.cat(
            (
                pad_to_length(chosen_ids, max_length, pad_value),
                pad_to_length(reject_ids, max_length, pad_value),
            ),
            dim=0,
        )
        max_length = max(c_mask.shape[1], r_mask.shape[1])
        max_length = get_padding_length(sp_enabled, tp_size, max_length)
        att_masks = torch.cat((pad_to_length(c_mask, max_length, 0), pad_to_length(r_mask, max_length, 0)), dim=0)
        return inputs_ids, att_masks

    def score_and_return_on_last_stage(self, data):
        """Function for just scoring.
        Arguments:
            tokens: prompt tokens extended to be of size [b, max_prompt_length]
        Note: Outside of model, other parameters only need to be available on
              rank 0.
        Outputs:
            output_log_probs: log probability of the selected tokens. size: [b, s]
        """

        # Log probability of the sequence (prompt + generated tokens).
        output_log_probs = None

        # =============
        # Run infernece
        # =============
        with torch.no_grad():
            if get_args().trainer_engine == TrainerEngine.DPO:
                chosen_ids = to_device("cuda", data["chosen"]).squeeze(1)
                rejected_ids = to_device("cuda", data["rejected"]).squeeze(1)
                chosen_mask = to_device("cuda", data["chosen_mask"]).squeeze(1)
                rejected_mask = to_device("cuda", data["rejected_mask"]).squeeze(1)
                prompt_id_lens = to_device("cuda", data["prompt_id_lens"])
                prompt_id_lens = torch.cat([prompt_id_lens, prompt_id_lens], dim=0)

                all_tokens, attn_masks = self.concatenated_inputs(chosen_ids, chosen_mask, rejected_ids, rejected_mask)
                ref_nll = self.score_reference_dpo(all_tokens, prompt_id_lens, attn_masks)
                if mpu.is_pipeline_last_stage():
                    output_log_probs = [ref_nll[:chosen_ids.shape[0]], ref_nll[chosen_ids.shape[0]:]]

            elif get_args().trainer_engine == TrainerEngine.ONLINE_DPO:
                ref_nll = self.score_reference_dpo(to_device("cuda", data["all_tokens"]))
                if mpu.is_pipeline_last_stage():
                    output_log_probs = -ref_nll # pylint: disable=invalid-unary-operand-type
            elif get_args().trainer_engine in (TrainerEngine.RLHF, TrainerEngine.GRPO):
                tokens = to_device("cuda", data["all_tokens"])

                attention_mask, position_ids = get_ltor_masks_and_position_ids_rlhf(tokens)

                # logits will be meanigful only in the last pipeline stage.
                logits = forward_step_helper(self.model, tokens, position_ids, attention_mask)

                if not self._parallel_output:
                    if mpu.is_pipeline_last_stage():
                        # Always the last stage should have an output.
                        assert logits is not None
                        assert logits.size(1) == tokens.size(1), "head(hidden(token))"
                        log_probs = F.log_softmax(logits, dim=2)

                        # Pick the tokens that we need to get the log
                        # probabilities for. Note that next input token is
                        # the token which we selected in the current logits,
                        # so shift by 1.
                        indices = torch.unsqueeze(tokens[:, 1:], 2)
                        output_log_probs = torch.gather(log_probs, 2, indices).squeeze(2)
                else:
                    if mpu.is_pipeline_last_stage():
                        vocab_parallel_logits = logits
                        logits_max = torch.max(vocab_parallel_logits, dim=-1)[0]
                        torch.distributed.all_reduce(logits_max,
                                                    op=torch.distributed.ReduceOp.MAX,
                                                    group=mpu.get_tensor_model_parallel_group())
                        logits.sub_(logits_max.unsqueeze(dim=-1))
                        # Get the partition's vocab indecies
                        get_vocab_range = VocabUtility.vocab_range_from_per_partition_vocab_size
                        partition_vocab_size = vocab_parallel_logits.size()[-1]
                        rank = mpu.get_tensor_model_parallel_rank()
                        world_size = mpu.get_tensor_model_parallel_world_size()
                        vocab_start_index, vocab_end_index = get_vocab_range(
                            partition_vocab_size, rank, world_size)

                        indices = torch.unsqueeze(tokens, 2)

                        # Create a mask of valid vocab ids (1 means it needs to be masked).
                        target_mask = (indices < vocab_start_index) | (
                            indices >= vocab_end_index)  # [b,s] 1 for not in range action, 0 for in range

                        masked_actionids = indices - vocab_start_index  # [b,s]
                        # Pick the tokens that we need to get the log
                        # probabilities for. Note that next input token is
                        # the token which we selected in the current logits,
                        # so shift by 1.
                        masked_actionids[:, 0, :] = 0
                        masked_actionids[target_mask] = 0  # [b,s]
                        logits_2d = vocab_parallel_logits.view(-1, partition_vocab_size)  # [n vp]
                        masked_actionids_1d = masked_actionids.view(
                            -1)  # [n] 0 for not in vocab range, target id -start for in range
                        arange_1d = torch.arange(start=0, end=logits_2d.size()[0],
                                                device=logits_2d.device)
                        predicted_logits_1d = logits_2d[
                            arange_1d, masked_actionids_1d]  # [n] in range target logit, not in range logits[0]
                        predicted_logits_1d = predicted_logits_1d.clone().contiguous()
                        action_logits = predicted_logits_1d.view_as(indices)
                        action_logits[target_mask] = 0.0  # [b s] 0 for not in range, logit for in range
                        # All reduce is needed to get the chunks from other GPUs.
                        torch.distributed.all_reduce(action_logits,
                                                    op=torch.distributed.ReduceOp.SUM,
                                                    group=mpu.get_tensor_model_parallel_group())
                        # Sum of exponential of logits along vocab dimension across all GPUs.
                        exp_logits = vocab_parallel_logits  # [ b, s, vp ]
                        torch.exp(vocab_parallel_logits, out=exp_logits)
                        sum_exp_logits = exp_logits.sum(dim=-1)
                        torch.distributed.all_reduce(sum_exp_logits,
                                                    op=torch.distributed.ReduceOp.SUM,
                                                    group=mpu.get_tensor_model_parallel_group())
                        log_probs = action_logits.squeeze(2) - torch.log(
                            sum_exp_logits + 1e-10)  # log ( exp(l) / sum(exp(li)

                        # shift by 1
                        output_log_probs = log_probs[:, 1:]
                        output_log_probs = output_log_probs.contiguous()

                        assert not torch.isnan(output_log_probs).any(), f"just out ref_logprobs {output_log_probs}"
                        assert output_log_probs.size(1) == tokens.size(1) - 1, "all token logprob except first one [1:]"
            else:
                raise RuntimeError(f"unexpected trainer_engine {get_args().trainer_engine}, expect one of {list(TrainerEngine)}")
        return output_log_probs

    def forward_step(self, data, iteration=None):
        '''

        ChatLearn calling
        chatlearn framework source:         ref_output = self.reference.forward_step(policy_output[0])

        :param data: global batch??? micro_batch?
        :return:
        '''
        if get_args().trainer_engine == TrainerEngine.DPO:
            ref_logprobs = self.score_and_return_on_last_stage(data)
            ref_out_dict = {
                "chosen": data["chosen"],
                "chosen_mask": data["chosen_mask"],
                "rejected": data["rejected"],
                "rejected_mask": data["rejected_mask"],
                "prompt_id_lens": data["prompt_id_lens"],
            }
            if mpu.is_pipeline_last_stage(): # for the last pipeline stage, ref_logprobs is a list
                ref_out_dict.update({
                    "reference_chosen_logps": ref_logprobs[0],
                    "reference_rejected_logps": ref_logprobs[1]
                })
            else: # for non-last pipeline stage, ref_logprobs is None
                ref_out_dict.update({
                    "reference_chosen_logps": None,
                    "reference_rejected_logps": None
                })
            return ref_out_dict
        else:
            ref_logprobs = self.score_and_return_on_last_stage(data)
            return {"ref_logprobs": ref_logprobs}
