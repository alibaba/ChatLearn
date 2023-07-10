import torch
from megatron import get_args
from megatron import get_tokenizer
from megatron.core import tensor_parallel
from megatron.model import GPTModel
from megatron.model.module import MegatronModule
from megatron.model.utils import get_linear_layer


def preprocess(prompt, response, tokenizer, max_length):
    prompt_chunk = []
    completion_chunk = []
    enc_chunk = []

    prompt_chunk.extend(prompt)
    completion_chunk.extend(response)
    completion_chunk.append(tokenizer.eod)

    # output prompt length
    prompt_length = len(prompt_chunk)
    completion_length = len(completion_chunk)
    if prompt_length > max_length and completion_length > max_length:
        # 都超长的时候，往中间靠
        prompt_chunk = prompt_chunk[-(max_length // 2):]
        completion_chunk = completion_chunk[:max_length // 2]
        enc_chunk = prompt_chunk + completion_chunk
        prompt_length = len(prompt_chunk)
        completion_length = len(completion_chunk)
        all_length = len(enc_chunk)
    elif prompt_length + completion_length > max_length:
        # 尽量保留prompt，仅在单独prompt就超过最大长度时候，向左边截断
        if prompt_length > max_length:
            enc_chunk = prompt_chunk + completion_chunk
            enc_chunk = enc_chunk[-max_length:]
            completion_length = len(completion_chunk)
            prompt_length = max_length - completion_length
        # completion超长或者两者和超长的时候，都从completion做截断
        else:
            enc_chunk = prompt_chunk + completion_chunk
            enc_chunk = enc_chunk[:max_length]
            prompt_length = len(prompt_chunk)
            completion_length = max_length - prompt_length
        all_length = len(enc_chunk)
    else:
        # 其他情况下保留原来的逻辑
        # padding to the last
        padding_length = max_length - prompt_length - completion_length
        padding_chunk = [tokenizer.eod] * (padding_length)
        enc_chunk = prompt_chunk + completion_chunk + padding_chunk
        all_length = len(enc_chunk) - len(padding_chunk)
    assert len(enc_chunk) == max_length
    assert completion_length + prompt_length <= max_length

    return {'ids': enc_chunk, 'length': all_length}


def batch_padded_tokenize_data(list_strs, tokenizer, max_length):
    processed_dict = [preprocess(tokenizer.tokenize(line[0]), tokenizer.tokenize(line[1]), tokenizer, max_length) for
                      line in list_strs]
    input_ids, input_lengths = [], []
    for item in processed_dict:
        input_ids.append(torch.tensor(item['ids']))
        input_lengths.append(item['length'])
    max_l = min(max(input_lengths), max_length)
    input_ids = torch.stack(input_ids, dim=0)[:, :max_l]
    input_eos_tok = torch.tensor(input_lengths) - 1

    return input_ids, input_eos_tok


class LinearPooler(MegatronModule):
    """Pooler layer.

    Pool hidden states of a specific token (for example start of the
    sequence) and add a linear transformation followed by a tanh.

    Arguments:
        hidden_size: hidden size
        init_method: weight initialization method for the linear layer.
            bias is set to zero.
    """

    def __init__(self, hidden_size, init_method, score_dimensions):
        super(LinearPooler, self).__init__()
        args = get_args()
        self.dense1 = get_linear_layer(hidden_size, hidden_size, init_method)
        self.dense2 = get_linear_layer(hidden_size, score_dimensions, init_method)
        self.sequence_parallel = args.sequence_parallel

    def forward(self, hidden_states, sequence_indices):
        # hidden_states: [s, b, h]
        # sequence_index: index of the token to pool.

        # gather data along sequence dimensions
        # same pooler is run on all tensor parallel nodes
        if self.sequence_parallel:
            hidden_states = tensor_parallel.gather_from_sequence_parallel_region(
                hidden_states,  # [s, b, h]
                tensor_parallel_output_grad=False)

        selected_hidden = torch.index_select(hidden_states, 0, sequence_indices)
        selected_hidden = selected_hidden.diagonal(dim1=0, dim2=1).T
        pooled = self.dense2(torch.nn.functional.relu(self.dense1(selected_hidden)))

        return pooled


class RewardModel(GPTModel):

    def __init__(self,
                 num_tokentypes=0,
                 parallel_output=True,
                 pre_process=True,
                 post_process=True,
                 pooler_head=LinearPooler,
                 score_dimension=1):
        args = get_args()
        super().__init__(num_tokentypes=num_tokentypes,
                         parallel_output=parallel_output,
                         pre_process=pre_process,
                         post_process=post_process)

        # self.parallel_output = parallel_output  # deprecated for now

        # self.pre_process = pre_process
        # self.post_process = post_process
        # self.fp16_lm_cross_entropy = args.fp16_lm_cross_entropy

        # self.language_model, self._language_model_key = get_language_model(
        #     num_tokentypes=num_tokentypes,
        #     add_pooler=False,
        #     encoder_attn_mask_type=AttnMaskType.causal,
        #     init_method=init_method_normal(args.init_method_std),
        #     scaled_init_method=scaled_init_method_normal(args.init_method_std,
        #                                                  args.num_layers),
        #     pre_process=self.pre_process,
        #     post_process=self.post_process)

        self.pooler_head = pooler_head(self.language_model.hidden_size, self.language_model.init_method,
                                       score_dimensions=score_dimension)
        self._pooler_head_key = 'pooler_head'
        self.tokenizer = get_tokenizer()
        # if not args.untie_embeddings_and_output_weights:
        #     self.initialize_word_embeddings(init_method_normal)

    def forward(self, input_ids=None, position_ids=None, attention_mask=None,
                ret_input_ids=None, ret_position_ids=None, ret_attn_mask=None,
                labels=None, tokentype_ids=None, inference_params=None,
                pooling_sequence_index=None,  ## add args
                list_strs=None,  ## add args
                ):
        # if list_strs is not None:
        #     args = get_args()
        #     assert input_ids is None
        #     input_ids, pooling_sequence_index = batch_padded_tokenize_data(list_strs, self.tokenizer)
        #     input_ids = input_ids.cuda()
        #     pooling_sequence_index = pooling_sequence_index.cuda()
        #     attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
        #                                                                     input_ids,
        #                                                                     self.tokenizer.vocab['<sep>'],
        #                                                                     args.reset_position_ids,
        #                                                                     args.reset_attention_mask,
        #                                                                     args.eod_mask_loss)

        lm_output = self.language_model(
            input_ids,
            position_ids,
            attention_mask,
            ret_input_ids=ret_input_ids,
            ret_position_ids=ret_position_ids,
            ret_attn_mask=ret_attn_mask,
            inference_params=inference_params)

        pooled_output = self.pooler_head(lm_output, pooling_sequence_index)

        return pooled_output  # [b x score_dim]

    def state_dict_for_save_checkpoint(self, prefix='', keep_vars=False):
        # state_dict_ = {}
        # state_dict_[self._language_model_key] \
        #     = self.language_model.state_dict_for_save_checkpoint(
        #         prefix=prefix, keep_vars=keep_vars)
        # # Save word_embeddings.
        # if self.post_process and not self.pre_process:
        #     state_dict_[self._word_embeddings_for_head_key] \
        #         = self.word_embeddings.state_dict(prefix=prefix,
        #                                           keep_vars=keep_vars)

        state_dict_ = super().state_dict_for_save_checkpoint(prefix, keep_vars)

        state_dict_[self._pooler_head_key] \
            = self.pooler_head.state_dict_for_save_checkpoint(
            prefix=prefix, keep_vars=keep_vars)

        return state_dict_

    def load_state_dict(self, state_dict, strict=True):
        """Customized load."""
        super().load_state_dict(state_dict, strict)

        # # Load word_embeddings.
        # if self.post_process and not self.pre_process:
        #     self.word_embeddings.load_state_dict(
        #         state_dict[self._word_embeddings_for_head_key], strict=strict)
        if self._pooler_head_key in state_dict:
            self.pooler_head.load_state_dict(state_dict[self._pooler_head_key], strict=strict)
        # if self._language_model_key in state_dict:
        #     state_dict = state_dict[self._language_model_key]
        # self.language_model.load_state_dict(state_dict, strict=strict)
