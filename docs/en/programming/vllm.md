# vLLM

ChatLearn support vLLM generation backend for intra-machine or inter-machine distirbuted inference. We enables to sync parameters between training and inference automatically, also allows to colocate training and inference model together.

For now, we enable vLLM to accelerate policy generation.

## Model Definition

Similar to inheriting `MegatronModule` for implementing [PolicyInference Model](../../../examples/megatron/models/old_policy_inference.py), the vLLM backend can be enabled by inheriting `VLLMModule` class and implementing the following key modules:
- model_provider: model definition function.
- setup: call model_provider to define model. Optionly, call `load_checkpoint` or others.
- build_dataset: Preprocess train/eval dataset with vLLM tokenizer.
- eval_forward: distributed inference tasks in eval mode.
- forward_step: distributed inference tasks in training mode.
- _add_request: prepare inputs for vLLM scheduler.
- decode_internal: decode generation outputs of vLLM as you need.

Code structure shows as following:

```python
from chatlearn import VLLMModule
from chatlearn.utils.vllm_utils import get_model, print_rank_0


class VLLMPolicyInference(VLLMModule):
    """Policy vLLM Inference"""

    def setup(self):
        pass

    def build_dataset(self, train_prompts, is_eval=False):
        pass

    def model_provider(self):
        """Build the model."""
        pass

    def eval_forward(self, data, iteration=0):
        pass

    def _add_request(self, data):
        pass

    def forward_step(self, data, iteration=0):
        pass

    def decode_internal(self, batched_outputs):
        pass
```

You can refer to[vllm_policy_inference.py](../../../examples/megatron/models/vllm_policy_inference.py), in which build_dataset/_add_request/forward_step/decode_internal clarified as following:

- build_dataset: Use `tokenizer`, you only need to return prompt_ids and prompt string. In `build_dataset`, [VLLMPromptPipeline](../../../examples/megatron/data/prompt_dataset.py#141) shows as following:
```python
class VLLMPromptPipeline(PromptPipeline):
    def __init__(self, prompts: List[str], max_prompt_length: int, tokenizer=None):

        for p in prompts:
            assert len(p) > 0, "Got empty prompt"
        assert max_prompt_length > 0, \
            "Prompt length for RLHF/OnlineDPO/GRPO trainer must be an integer greater than 0"

        # tokenizer prompts of self-defined format
        # only return prompt str and prompt ids
        self.prompts = [(prompt, tokenizer.encode(prompt[:max_prompt_length])) for prompt in prompts]
        self.prompts_ids = []
        for prompt, prompt_ids in self.prompts:
            p = {"input_ids": prompt_ids, "prompt": prompt}
            self.prompts_ids.extend([copy.deepcopy(p)])
        # set tokenizer
        self.tokenizer = tokenizer

class VLLMPolicyInference(VLLMModule):
    def build_dataset(self, train_prompts, is_eval=False):
        max_prompt_length = (
            self.model_args.get("seq_length") - self.model_args.get("max_new_tokens")
        )
        # TODO: read from files
        prompts_dataset = VLLMPromptPipeline(
            train_prompts, max_prompt_length, self.tokenizer.tokenizer)

        return prompts_dataset
```

- _add_request: add preprocessed request pairs (input_ids, prompt) to vLLM scheduler
```python
    def _add_request(self, data, is_eval=False):
        return self._add_request_internal(data["prompt"], data["input_ids"], is_eval=is_eval)
```

- forward_step: take batch `data` scheduled by vLLM scheduler as input, and call `execute_step` for distributed inference.

```python
    def _forward_step(self, data, iteration, eval_mode):
        assert iteration >= 0
        assert isinstance(eval_mode, bool)
        seq_group_metadata_list = data["seq_group_metadata_list"]
        blocks_to_swap_in = data["blocks_to_swap_in"]
        blocks_to_swap_out = data["blocks_to_swap_out"]
        blocks_to_copy = data["blocks_to_copy"]

        outputs = self.execute_step(
            seq_group_metadata_list, blocks_to_swap_in, blocks_to_swap_out, blocks_to_copy)

        return outputs

    def forward_step(self, data, iteration=0):
        return self._forward_step(data, iteration, eval_mode=False)
```

- decode_internal: Refer to [examples](../../../examples/megatron/models/vllm_policy_inference.py#L119) for more details. Format of param `batched_outputs` is List[RequestOutput], in which [RequestOutput](https://github.com/vllm-project/vllm/blob/v0.5.1/vllm/outputs.py#L67)includes the following key attributes:

|   Attibute  |Type| Comment  |
|:------:|:-----:|:-----:|
| request_id      | int                   | prompt request number   |
| prompt          | string                | prompt token string        |
| prompt_token_ids| List(int)             |prompt ids list      |
| prompt_logprobs | List(Dict(float))     | each prompt token has one relative logprob value |
| outputs         | List(CompletionOutput)| See the next table for details.|

vLLM `CompletionOutput` includes:

|   Attibute  |Type| Comment  |
|:------:|:-----:|:-----:|
| index              | int                 | response index, which helps to number different response for one prompt    |
| text               | string              | response token string        |
| token_ids          | List(int)           | list of generated response token ids      |
| cumulative_logprob | float               | logprob cumulative value of all generated tokens for current response |
| logprobs           |  List(Dict(float))  |  logprob value of each generated token|



## model configuration yaml

You can modify `model_config_file` in `rlhf.yaml`, For example:

```yaml
policy:
    model_config_file: vllm_policy_inference.yaml
    ...
```

Or you can refer to [llama2 model yaml](../../../examples/megatron/configs/llama2/vllm_rlhf.yaml).

## hyperparameter configuration yaml

Hyperparameter for vLLM can be divied into 5 parts:
- sampling params: sampling hyperparameter

|   Attibute  |Type| Comment  |
|:------:|:-----:|:-----:|
| n      | int| Number of responses for each prompt.    |
| ignore_eos              | bool  | Whether to stop generating tokens for prompt which has generated a eos token already   |
| top_p                   | float | Float that controls the cumulative probability of the top tokens to consider      |
| top_k                   | int   |Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens. |
| temperature             | float |  Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.|
| use_beam_search         |  bool |  Whether to use beam search instead of sampling. |
| eval_temperature        | float |  Like `temperature`, but for evaluation scene. |
| eval_top_k              |  int  |  Like `eval_top_k`, but for evaluation scene.|
| eval_top_p              | float |  Like `eval_top_p`, but for evaluation scene.|
| stop_token_list         | string|  stop token string, seperated by semicolon.|
| new_token_limit         |  bool |  Whether to limit the number of generated tokens.|
| prompt_logprobs         |  int  |  Whether to output logprobs for prompt token. Set `None` by default to save gpu memory. `1` to enable, `None` to disable.|


- scheduler config: hyperparameters for vLLM batch scheduling.

|   Attibute  |Type| Comment  |
|:------:|:-----:|:-----:|
| max_num_batched_tokens   | int| Upper bound of token number in a batch. Please set `batch_size*(max_seq_len-max_prompt_length)`.   |
| max_paddings             | int| Upper bound of padding token number in a batch.   |


- cache config: Hyperparameter to define vLLM cache blocks, relative to gpu/cpu memory usage.


|   Attibute  |Type| Comment  |
|:------:|:-----:|:-----:|
| block_size               | int   | gpu blocks size. Set `16` MB by default, you can infer by the largest activation size. |
| gpu_memory_utilization   | float | Upper bound of GPU memory ratio you can required for all processes when generating. Range as (0, 1.0] |
| swap_space               | int   | When GPU memory is limited, take CPU memory to swap in or swap out. Set `4` GB by default.   |
| sliding_window           | int   | Set `None` by default. vLLM don't support other settings.  |

- tokenizer: Repo to load vLLM tokenizer, which shows as [LLama2-7B-hf](https://huggingface.co/meta-llama/Llama-2-7b)

- Others: `includes` specifies model structure.


You can refer to [vLLM Hyperparameter Configuration](../../../examples/megatron/configs/llama2/vllm_policy_inference.yaml) for details.
