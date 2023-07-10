import torch
import torch.nn.functional as F
from megatron import get_args
from megatron import print_rank_0
from megatron.core import mpu
from megatron.core.tensor_parallel.utils import VocabUtility
from megatron.text_generation.communication import broadcast_from_last_to_first_pipeline_stage
from models.policy_model import PolicyModel
from utils.forward_step import forward_step_helper

from rlhf.utils import to_device
from .constants_ppo import get_ltor_masks_and_position_ids
from .old_policy_inference import PolicyMegatronInference

"""Sample Generate GPT"""


class PolicyReference(PolicyMegatronInference):

    def model_provider(self, pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0('building GPT model ...')
        args = get_args()

        if hasattr(args, "parallel_output") and args.parallel_output:
            self._parallel_output = True
            print_rank_0("enable parallel output")
        else:
            self._parallel_output = False
        model = PolicyModel(num_tokentypes=0, parallel_output=self._parallel_output, pre_process=pre_process,
                            post_process=post_process)

        return model

    def score_and_return_on_first_stage(self, model, tokens):
        """Function for just scoring.
        Arguments:
            model: no interleaving is supported.
            tokens: prompt tokens extended to be of size [b, max_prompt_length]
            lengths: original prompt length, size: [b]
        Note: Outside of model, other parameters only need to be available on
              rank 0.
        Outputs:
            output_log_probs: log probability of the selected tokens. size: [b, s]
        """

        args = get_args()

        batch_size = tokens.size(0)
        all_tokens_len = tokens.size(1)
        max_sequence_length = min(all_tokens_len, args.max_position_embeddings)

        # Log probability of the sequence (prompt + generated tokens).
        output_log_probs = None
        output_log_probs_size = (batch_size, max_sequence_length - 1)

        # =============
        # Run infernece
        # =============
        with torch.no_grad():
            attention_mask, position_ids = get_ltor_masks_and_position_ids(tokens)

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
                    # print(f"megatron score: log_probs size: {log_probs.size()}")
                    # print(f"megatron score: indices size: {indices.size()}")
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

        # ======================================
        # Broadcast to the first pipeline stage.
        # ======================================

        output_log_probs = broadcast_from_last_to_first_pipeline_stage(
            output_log_probs_size, torch.float32, output_log_probs)

        return output_log_probs

    def forward_step(self, data):
        '''

        RLHF calling
        rlhf framework source:         ref_output = self.reference.forward_step(policy_output[0])

        :param data: global batch??? micro_batch?
        :return:
        '''
        all_tokens = to_device("cuda", data["all_tokens"])
        ref_logprobs = self.score_and_return_on_first_stage(self.model, all_tokens)

        assert not torch.isnan(ref_logprobs).any(), f"just out ref_logprobs {ref_logprobs}"

        assert ref_logprobs.size(1) == all_tokens.size(1) - 1, "all token logprob except first one [1:]"

        return {"ref_logprobs": ref_logprobs}
