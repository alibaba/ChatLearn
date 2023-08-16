import torch
from megatron import get_args, get_tokenizer
from megatron import print_rank_0
from megatron.core import mpu
from megatron.text_generation.communication import broadcast_from_last_to_first_pipeline_stage
from megatron.training import get_model
from models.value_model import ValueModel

from chatlearn import RLHFMegatronModule
from chatlearn.utils import to_device
from .constants_ppo import get_ltor_masks_and_position_ids
from .forward_step import forward_step_helper


class ValueMegatronInference(RLHFMegatronModule):

    def setup(self):
        self.buffer = {}
        self.stats = {}
        self.tokenizer = get_tokenizer()
        self.args = get_args()
        # Set up model and load checkpoint
        model = get_model(self.model_provider, wrap_with_ddp=False)

        assert len(model) == 1, "Above condition should have caught this"
        self.model = model[0]
        self.model.eval()
        return 'ok'

    def model_provider(self, pre_process=True, post_process=True):
        """Build the model."""

        print_rank_0('building GPT model ...')
        model = ValueModel(num_tokentypes=0, parallel_output=False, pre_process=pre_process, post_process=post_process,
                           stats=self.stats, buffer=self.buffer)

        return model

    def forward_step(self, data_b, iteration):
        '''
        :param data_b: micro_batch??
        :return:
            {"old_values": output_values}
        '''
        all_tokens = to_device("cuda", data_b["all_tokens"])

        args = get_args()

        batch_size = all_tokens.size(0)
        max_length = all_tokens.size(1)
        max_sequence_length = min(max_length, args.max_position_embeddings)

        # Log probability of the sequence (prompt + generated tokens).
        output_values = None
        output_values_size = (batch_size, max_sequence_length)

        # =============
        # Run infernece
        # =============
        with torch.no_grad():
            attention_mask, position_ids = get_ltor_masks_and_position_ids(
                all_tokens)
            # logits will be meanigful only in the last pipeline stage.
            lm_output = forward_step_helper(self.model, all_tokens, position_ids, attention_mask, pooling=True)

            if mpu.is_pipeline_last_stage():
                # Always the last stage should have an output.
                assert lm_output is not None
                values = lm_output
                # Pick the tokens that we need to get the log
                # probabilities for. Note that next input token is
                # the token which we selected in the current logits,
                # so shift by 1.
                output_values = values

        # ======================================
        # Broadcast to the first pipeline stage.
        # ======================================
        output_values = broadcast_from_last_to_first_pipeline_stage(
            output_values_size, torch.float32, output_values)

        return {"old_values": output_values}
