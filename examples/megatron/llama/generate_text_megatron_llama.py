# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
import torch
from megatron import get_args
from megatron.initialize import initialize_megatron
from megatron.model.gpt_model import GPTModel
from megatron.training import get_model

try:
    from megatron.model import ModelType
except ImportError:
    from megatron.core.enums import ModelType
from megatron.checkpointing import load_checkpoint
from megatron.text_generation.api import generate_and_post_process


def get_tasks_args(parser):
    group = parser.add_argument_group(title='text generation')

    group.add_argument('--text-generate-input-file', type=str, default='')
    group.add_argument('--text-generate-output-file', type=str, default='')
    group.add_argument(
        '--input-len',
        type=int,
        default=1,
        help='input lenth for measure end to end text generation average time')
    group.add_argument('--top-p',
                       type=float,
                       default=0.0,
                       help='Top p sampling.')
    group.add_argument('--top-k', type=int, default=0, help='Top k sampling.')

    group.add_argument('--out-seq-length',
                       type=int,
                       default=1024,
                       help='Size of the output generated text.')

    group.add_argument('--temperature',
                       type=float,
                       default=1.0,
                       help='Sampling temperature.')
    group.add_argument('--extra-vocab-size',
                       type=int,
                       default=1,
                       help='--extra-vocab-size')
    return parser


def model_provider(pre_process=True, post_process=True):
    args = get_args()
    args.model_type = ModelType.encoder_or_decoder
    model = GPTModel(num_tokentypes=0,
                     parallel_output=True,
                     pre_process=pre_process,
                     post_process=post_process)
    return model


if __name__ == '__main__':
    initialize_megatron(extra_args_provider=get_tasks_args)

    args = get_args()

    model = get_model(model_provider,
                      model_type=ModelType.encoder_or_decoder,
                      wrap_with_ddp=False)
    assert args.load is not None
    if args.load is not None and args.no_load_optim:
        load_checkpoint(model, None, None)
    torch.distributed.barrier()

    if not isinstance(model, list):
        model = [model]

    assert len(model) == 1, 'Above condition should have caught this'
    model = model[0]
    prompts = ["I have a dream", "Write a warm family story for me.", "杭州的旅行攻略", "write a meeting schedule"]
    num_examples = len(prompts)

    pred_outputs = []

    buffer = []

    for idx, prompt in enumerate(prompts):
        prompt = prompt[:args.seq_length]
        if len(buffer) < args.micro_batch_size:
            buffer.append(prompt)
        if len(
            buffer
        ) == args.micro_batch_size or idx == num_examples - 1:
            sl = args.out_seq_length
            tk = args.top_k
            tp = args.top_p
            temperature = args.temperature
            prompts_plus_generations, _, _, _ = \
                generate_and_post_process(model,
                                          prompts=buffer,
                                          tokens_to_generate=sl,
                                          top_k_sampling=tk,
                                          temperature=temperature,
                                          top_p_sampling=tp)

            for prompt, p_and_g in zip(buffer,
                                       prompts_plus_generations):
                generation = p_and_g.replace('<|endoftext|>', '')
                print(p_and_g)
                pred_outputs.append(generation)
            buffer.clear()

        if idx % args.micro_batch_size == 0:
            print('processed {} examples'.format(idx))
