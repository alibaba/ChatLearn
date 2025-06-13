进阶配置
========

vLLM
-----
ChatLearn supports vLLM for distributed reasoning across machines, and supports automatic parameter synchronization between vLLM and the training backend.
Examples of vLLM usage can be found in `vllm_policy_inference.py <https://github.com/alibaba/ChatLearn/blob/main/examples/megatron/models/vllm_policy_inference.py>`_.

YAML Configuration
>>>>>>>>>>>>>>>>>>

The hyperparameters for vLLM can be divided into the following sections:

- sampling params: Sampling hyperparameters, with specific meanings as shown in the table below

.. csv-table::
   :header: "Attribute", "Type", "Description"

   "n",               "int",      "Number of responses for each prompt output"
   "ignore_eos",               "bool",      "Control whether to end generation when generating eos tokens for a certain prompt"
   "top_p",               "float",      "Float that controls the cumulative probability of the top tokens to consider"
   "top_k",               "int",      "Integer that controls the number of top tokens to consider. Set to -1 to consider all tokens."
   "temperature",               "float",      "Float that controls the randomness of the sampling. Lower values make the model more deterministic, while higher values make the model more random. Zero means greedy sampling."
   "use_beam_search",               "bool",      "Whether to use beam search instead of sampling."
   "eval_temperature",               "float",      "Same as temperature, but used in the Evaluation scenario."
   "eval_top_k",               "int",      "Same as top_k, but used in the Evaluation scenario."
   "eval_top_p",               "float",      "Same as top_p, but used in the Evaluation scenario."
   "stop_token_list",               "string",      "Stop token string, separated by semicolon."
   "new_token_limit",               "bool",      "Whether to limit the number of generated tokens"
   "prompt_logprobs",               "int",      "Prompt token logprobs calculation, set to None to save memory"

- scheduler config: Configuration hyperparameters for batch scheduling of data samples

.. csv-table::
   :header: "Attribute", "Type", "Description"

   "max_num_batched_tokens",               "int",      "Upper limit on the number of tokens in a batch of data, recommended to be set as batch_size*(max_seq_len-max_prompt_length)"
   "max_paddings",               "int",      "Upper limit on the number of padding tokens in a batch of data"

- cache config: Configuration for generating vLLM cache blocks, related to GPU/memory usage

.. csv-table::
   :header: "Attribute", "Type", "Description"

   "block_size",               "int",      "GPU blocks size, default is 16MB, can be derived based on the activation size of the specific model"
   "gpu_memory_utilization",               "float",      "Set the upper limit of GPU memory utilization for all processes during reasoning, range (0, 1.0], higher is better when GPU memory is sufficient"

- tokenizer: Directory for vLLM tokenizer reading, can refer to `LLama2-7B-hf <https://huggingface.co/meta-llama/Llama-2-7b>`_
- Others: includes specifying model structure and other parameters; refer to `vLLM hyperparameter configuration <https://github.com/alibaba/ChatLearn/blob/main/examples/megatron/configs/llama2/vllm_policy_inference.yaml>`_.


StreamDataset
-------------

`StreamDataset` 接收 `Env` rollout 产生的数据，并重组 batch 提供给 Alignment 训练模块 `Trainer`。目前我们支持两种形式的 `StreamDataset`:

1. `fixed` ：这种形式生成的总训练样本数是由配置 `sample_per_episode` 指定的。`Env` 接收 `sample_per_episode` 个 prompts，生成 `sample_per_episode` 个训练样本。`Trainer` 接受 `sample_per_episode` 个训练样本进行训练。
2. `dynamic` : 这种形式生成的总训练样本数是动态判断的。`Env` 接收 `sample_per_episode` 个 prompts，生成 `N*sample_per_episode` 个训练样本，这里 `N>0`。`Trainer` 接受 `N*sample_per_episode` 个训练样本进行训练。

YAML 配置
>>>>>>>>>

.. code-block:: yaml

    runtime:
        # one of ["fixed", "dynamic"]
        stream_data_loader_type: fixed
        #: max number of replay episodes, if `max_replay_episode` is set to -1, then replay all episodes
        #: if `max_replay_episode` is set to 0, then replay is disabled
        max_replay_episode: int = 0
        #: replay after n episodes
        replay_episode_offset: int = 0


.. csv-table::
   :header: "参数名", "类型", "注释"

   "stream_data_loader_type",               "str",      "指定类型，默认是 fixed，必须是以下类型之一，['fixed', 'dynamic']"
   "max_replay_episode",               "int",      "指定 replay 的最近的 max_replay_episode 个 episode，超过 max_replay_episode，会淘汰最老的 episode 数据。如果 max_replay_episode 设为 -1，则不会淘汰，记录每个 episode 的历史数据。如果 max_replay_episode 设为 0，则不会开启 replay。"
   "replay_episode_offset",               "int",      "指定从第replay_episode_offset+1个episode开始replay，记录episode 的历史数据。默认为0。"



replay_sample_fn
>>>>>>>>>>>>>>>

`replay_sample_fn` 是用户自定义的 replay buffer sample 函数。

.. code-block:: python

    def replay_sample_fn(episode_replay_buffers) -> List[dict]:
        """
        Args:
            episode_replay_buffers : List[EpisodeReplayBuffer]
        Return: list of dict
        """


`replay_sample_fn` 接收 `episode_replay_buffers`，`episode_replay_buffers` 是一个 list 的 `EpisodeReplayBuffer`。每个 `EpisodeReplayBuffer` 记录了一个 episode 的 samples。`EpisodeReplayBuffer` 有两个关键属性：

1. `episode_id` 记录了当前是第几个 episode
2. `buffer` 记录了所有的 samples，类型是 `List[dict]`，每个 dict 是一个 sample。

通过 `engine.set_replay_sample_fn(replay_sample_fn)` 用户可以设定自定义的 `replay_sample_fn` 。

示例
>>>>

下面这个示例将所有的 `episode_replay_buffers` 中的 sample 合并起来，返回一个多个 episode 完整的 sample 数据。

.. code-block:: python

    def replay_sample_fn(episode_replay_buffers):
        buffers = []
        for replay_buffer in episode_replay_buffers:
            buffers += replay_buffer.buffer
        # episode_id = episode_replay_buffers[-1].episode_id
        return buffers

    engine = RLHFEngine(policy, reference, reward, value, ppo_policy, ppo_value)
    engine.set_replay_sample_fn(replay_sample_fn)

LoRA
----

LoRA 是 Parameter Efficient 的方法之一。已有研究表明了过度参数化的模型其实是位于一个低的内在维度上，所以 LoRA 作者假设在模型适应过程中的权重变化也具有较低的“内在等级”。LoRA 的主要方法为冻结一个预训练模型的矩阵参数 `W` ，并选择用重新初始化的小矩阵 `A` 和 `B` （类 SVM）来替代，在下游任务时只更新 `A` 和 `B`，其中 `W` 的 shape 为 `[d, k]` , `A/B` 的shape分别为 `[d, r]、[r, k]` 。⚠️收敛需要调整 `learning rate` 、LoRA 等相关参数，LoRA 使用及其参数介绍如下。

YAML 配置
>>>>>>>>>>>>>>>>

以下为配置打开 LoRA 的示例，用户可以在某个模型配置中加入 `lora` 配置段，通过 `enable_lora: True` 打开LoRA，同时设置 `lora_dim`, `lora_layer` 等参数。关于 LoRA 配置项的 API 详见 :ref:`lora-config`.

.. code-block:: yaml

    models:
        ppo_policy:
            model_config_file: ppo_policy.yaml
            trainable: True
            lora:
              enable_lora: True
              lora_dim: 64
              lora_layer: ColumnParallelLinear,LinearLayer,RowParallelLinear
              lora_dropout: 0.05

代码示例
>>>>>>>>>

下面的示例展示了如何设置模型的 LoRA 优化。如果用户在 yaml 中配置了 `enable_lora: True`，则需在模型定义完成后, 接入完成LoRA 转化函数 `convert_layer_to_lora`，如下：

.. code-block:: python

    from chatlearn.models.megatron.lora import convert_layer_to_lora
    model = PolicyModel()
    if self.module_args.lora.enable_lora:
        model = convert_layer_to_lora(model)

Batch generation 优化
---------------------

默认配置中，推理阶段的每 episode 中数据一般进行了随机 shuffle，导致 Batch 内样本的 prompt_len 分布不一，在 batch generation 过程中会将所有 prompt padding 到 batch 内最长，增加了大量无效计算。一个优化方式是可按 prompt length 预先排序，降低无效 padding 的 tokens 占比。Prompt generation 阶段可分为以下两步：

1. initiation：选择 batch 内 `min_prompt_len`，一次性输入 `[batch_size, min_prompt_len, hidden_size]` 的特征向量进行推理，生成下一个 token；
2. increment：基于 initiation 输出的 token，循环输入上一个迭代输出的 token，直到生成 `<EOS>` 为结束。

如果对 prompt 进行排序，随着 batch 内 `min_prompt_len` 增加，我们观察到显存开销的提高，容易出现 OOM。通过设置 `min_prompt_length` 参数可以缓和显存问题，具体介绍如下。

YAML 配置
>>>>>>>>>

以下为配置打开 batch generation 优化的示例，用户可以在某个模型配置中加入 `batch_generation` 配置段，通过 `ranking: True` 打开。关于 `batch_generation` 配置项的 API 详见 :ref:`batch-generation-config`.

.. code-block:: yaml

    models:
        policy:
            model_config_file: policy_inference.yaml
            trainable: False
            batch_generation:
              ranking: True
              min_prompt_length: ${batch_generation_min_prompt_length:0}

