# vLLM

ChatLearn中支持vLLM进行跨机分布式推理，支持vllm和training backend之间的参数自动同步，同时支持training和generation复用相同资源和显存。

目前，ChatLearn 中Policy generation模型使用vLLM backend来节约显存占用，加速大模型推理任务，

## 模型定义

类似于继承`MegatronModule`实现[PolicyInference模型](https://github.com/alibaba/ChatLearn/blob/main/examples/megatron/models/old_policy_inference.py),PolicyInference模型若想基于vLLM后端完成generation，需要继承`VLLMModule`父类，实现以下关键模块：
- setup：调用model_provider定义模型，可根据需要决定是否load_checkpoint等。
- build_dataset：调用vLLM tokenizer处理数据，生成prompt dataset。
- forward_step：RLHF/OnlineDPO训练任务中调用完成分布式推理。
- decode_internal：根据实际需要，将vLLM输出的generation结果解析为相应格式。

代码结构参考如下：

```python
from chatlearn import VLLMModule
from chatlearn.utils.vllm_utils import get_model, print_rank_0


class VLLMPolicyInference(VLLMModule):
    """Policy vLLM Inference"""

    def setup(self):
        pass

    def build_dataset(self, train_prompts, is_eval=False):
        pass

    def forward_step(self, data, iteration=0):
        pass

    def decode_internal(self, batched_outputs):
        pass
```

示例可参考[vllm_policy_inference.py](https://github.com/alibaba/ChatLearn/blob/main/examples/megatron/models/vllm_policy_inference.py)，补充说明build_dataset、forward_step、decode_internal如下：

- build_dataset：调用tokenizer处理只需要返回prompt_ids、prompt str，其中build_dataset的[VLLMPromptPipeline](https://github.com/alibaba/ChatLearn/blob/main/examples/megatron/data/prompt_dataset.py#141)具体逻辑如下：
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

- forward_step：参数中data为vLLM scheduler调度的批数据，格式固定，调用vLLM的execute_step完成分布式推理

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

- decode_internal：可参考[examples](https://github.com/alibaba/ChatLearn/blob/main/examples/megatron/models/vllm_policy_inference.py#L119)实现。参数batched_outputs格式为List[RequestOutput]，其中[RequestOutput](https://github.com/vllm-project/vllm/blob/v0.5.1/vllm/outputs.py#L67)包含以下重要attributes：

|   属性  |类型| 含义  |
|:------:|:-----:|:-----:|
| request_id      | int| prompt request编号    |
| prompt          | string| prompt token string        |
| prompt_token_ids| List(int) |prompt ids list      |
| prompt_logprobs | List(Dict(float)) |每个prompt token对应的logprob value |
| outputs         | List(CompletionOutput)| 详见下表|

其中vLLM CompletionOutput类包含属性：

|   属性  |类型| 含义  |
|:------:|:-----:|:-----:|
| index              | int                 | response编号，用以区分同一prompt的不同回答    |
| text               | string              | response token string        |
| token_ids          | List(int)           | 生成的response token ids list      |
| cumulative_logprob | float               | 生成response的所有tokens的logprobs累计值求和 |
| logprobs           |  List(Dict(float))  |  生成的response中每个token对应的logprobs|


## 模型配置

可以直接修改 `rlhf.yaml`中policy模型的 `model_config_file` 配置，例如:

```yaml
policy:
    model_config_file: vllm_policy_inference.yaml
    ...
```
也可以参考示例 [llama2模型配置](https://github.com/alibaba/ChatLearn/blob/main/examples/megatron/configs/llama2/vllm_rlhf.yaml)。

## 超参配置

vLLM超参可分为五部分：
- sampling params：采样超参，具体含义如下表

|   属性  |类型| 含义  |
|:------:|:-----:|:-----:|
| n      | int| 每个prompt输出response的个数    |
| ignore_eos              | bool  | 控制某个prompt在生成eos tokens时是否结束生成   |
| top_p                   | float | Float that controls the cumulative probability of the top tokens to consider      |
| top_k                   | int   |Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens. |
| temperature             | float |  Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling.|
| use_beam_search         |  bool |  Whether to use beam search instead of sampling. |
| eval_temperature        | float |  和temperature一样，但在Evaluation场景使用。|
| eval_top_k              |  int  |  和top_k一样，但在Evaluation场景使用。|
| eval_top_p              | float |  和top_p一样，但在Evaluation场景使用。|
| stop_token_list         | string|  stop token string, seperated by semicolon.|
| new_token_limit         |  bool |  是否限制生成tokens数|
| prompt_logprobs         |  int  |  Prompt token计算logprobs，默认为了节省显存，设置为None，即不进行logprobs计算|


- scheduler config：数据样本批调度配置超参

|   属性  |类型| 含义  |
|:------:|:-----:|:-----:|
| max_num_batched_tokens   | int| 批数据的tokens数的上限，建议设置为batch_size*(max_seq_len-max_prompt_length)   |
| max_paddings             | int| 批数据中padding tokens数的上限   |


- cache config：生成vLLM cache blocks的配置，与显存/内存使用有关


|   属性  |类型| 含义  |
|:------:|:-----:|:-----:|
| block_size               | int   | gpu blocks size，默认为16MB，可根据具体模型的activation size推导  |
| gpu_memory_utilization   | float | 设置推理过程中所有进程的显存使用上限占比，范围(0, 1.0]，在显存充足时，上限越高越好 |
| swap_space               | int   | 在GPU显存不足时，和CPU换入换出的内存大小，单位GB   |
| sliding_window           | int   | 默认为None，vLLM暂不支持设置。  |

- tokenizer：vLLM tokenizer读取目录，可参考[LLama2-7B-hf](https://huggingface.co/meta-llama/Llama-2-7b)
- 其他：includes指定模型结构等其余参数；

可以参考 [vLLM超参配置](https://github.com/alibaba/ChatLearn/blob/main/examples/megatron/configs/llama2/vllm_policy_inference.yaml)。
