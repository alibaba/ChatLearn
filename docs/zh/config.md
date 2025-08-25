# 配置说明

ChatLearn 的配置主要由两个部分组成：
- `runtime_args`：框架的核心训练配置。
- `models`：每个独立模型的配置。

配置模板位于 `ChatLearn/template` 目录中。

---

## runtime_args

```yaml
runtime_args:
  # setup配置
  train_backend: fsdp
  rollout_backend: vllm
  exp_name: grpo_fsdp
  colocation: [policy,policy_trainer,ref_policy]
  # 路径配置
  output_dir: your_output_dir
  data_path: your_data_path
  eval_data_path: your_eval_data_path
  data_checkpoint_path: ${runtime_args.output_dir}/data_checkpoint_path/
  # 训练配置
  num_episode: 200
  sample_per_episode: 512
  train_global_batch_size: 512
  save_episode_interval: 200
  # 数据配置
  data_shuffle: True
  data_rerank: True
  # 评估配置
  eval_episode_interval: 5
  enable_eval_before_training: False
  log_args_dict:
    log_dir: ${runtime_args.output_dir}
    enable_wandb: False
    wandb_project: your_wandb_project
    wandb_dir: ${runtime_args.output_dir}
    wandb_name: ${runtime_args.exp_name}
    wandb_id: ${runtime_args.exp_name}
    wandb_resume: allow
```

- `runtime_args.train_backend`：训练后端，支持 **fsdp** 或 **megatron**。
- `runtime_args.rollout_backend`：推理后端，可选择 **vllm** 或 **sglang**。
- `runtime_args.exp_name`：实验名称，用于日志记录。
- `runtime_args.colocation`：列出的模型将放置在同一个在 GPU 上并顺序执行, 当执行一个模型时，其他模型权重会卸载到内存中。
- `runtime_args.output_dir`：保存所有中间训练结果的目录。
- `runtime_args.data_path`：训练数据路径。请确保该目录下的数据文件与数据读取代码兼容。
- `runtime_args.eval_data_path`：评估数据路径。请确保该目录下的数据文件与数据读取代码兼容。
- `runtime_args.data_checkpoint_path`：数据检查点保存路径。默认为 `runtime_args.output_dir` 下的 `data_checkpoint_path/`。
- `runtime_args.num_episode`：总训练轮数（episode）。每轮包含多次权重更新。
- `runtime_args.sample_per_episode`：每轮训练的样本数量（`sample_per_episode = prompt_per_episode * num_inference_per_prompt`）。
- `runtime_args.train_global_batch_size`：将 `sample_per_episode` 划分为多个train_global_batch_size。每个批次用于一次模型权重更新。
- `runtime_args.save_episode_interval`：保存中间检查点的间隔。
- `runtime_args.eval_episode_interval`：执行评估的轮数间隔。
- `runtime_args.data_shuffle`：若启用，则打乱数据集样本顺序，忽略原始顺序。
- `runtime_args.data_rerank`：若启用，则相同数据样本的多个副本不会被分配到同一个推理 actor。
- `runtime_args.enable_eval_before_training`：是否在训练前进行一次评估。
- `runtime_args.log_args_dict`：日志记录配置。当 `runtime_args.enable_wandb=True` 时，请确保已登录 wandb账号。

---

## models

对于 GRPO 算法，ChatLearn 使用四个模型：`policy_trainer`、`ref_policy`、`policy` 和 `reward`。  
*注意：`policy_trainer` 和 `ref_policy` 共享相同的训练后端。*

### policy_trainer

#### 通用配置

```yaml
policy_trainer:
  free_gpu_memory:
    offload_weights: True
    offload_optimizer_states: True
    free_grad_buffers: True
  optimizer:
    lr: 2e-6
    clip_grad: 1
  trainable: True
  generation_batch_size: 8
  train_micro_batch_size: ${runtime_args.train_micro_batch_size}
  packing: False
  max_token_in_packing: 32768
  load: your_hf_model_path
  pos_clip_ratio: 0.2
  neg_clip_ratio: 0.2
  entropy_coef: 0.0
  kl_coef: 0.0
  gpu_per_process: 1
  num_gpu: 1
```

- `models.policy_trainer.free_gpu_memory.*`：控制 GPU 内存卸载；在colocation场景中建议全部设为 `True`（目前不支持非colocation）。
- `models.policy_trainer.optimizer.lr`：学习率。
- `models.policy_trainer.optimizer.clip_grad`：梯度裁剪阈值。
- `models.policy_trainer.trainable`：是否启用训练，此模型应始终设为 `True`。
- `models.policy_trainer.generation_batch_size`：单次模型前向传播的batch_size大小（用于计算 **old_logprobs**）。
- `models.policy_trainer.train_micro_batch_size`：每个每个模型在训练时的batch_size大小，用于前向/反向传播和梯度累积。
- `models.policy_trainer.packing`：若启用，则样本将按总token数不超过 `models.policy_trainer.max_token_in_packing` 重新进行分组。分组后，每批将打包为单个序列传递给模型推理。启用后，`generation_batch_size` 和 `train_micro_batch_size` 将被忽略。
- `models.policy_trainer.max_token_in_packing`：当 `packing` 启用时用于重新分组的token上限。
- `models.policy_trainer.load`：模型权重的加载路径。
- `models.policy_trainer.pos_clip_ratio`, `models.policy_trainer.neg_clip_ratio`：GRPO 算法的裁剪系数。
- `models.policy_trainer.entropy_coef`：设为大于 0.0 则在训练过程中启用entropy loss。
- `models.policy_trainer.kl_coef`：设为大于 0.0 则在训练过程中启用kl loss。
- `models.policy_trainer.gpu_per_process`：每个 Ray actor 分配的 GPU 数量。
- `models.policy_trainer.num_gpu`：训练所用的总 GPU 数。

#### FSDP policy_trainer 配置

以下是 FSDP 训练后端的特定配置：

```yaml
policy_trainer: 
  fsdp_size: ${models.policy_trainer.num_gpu}
  ulysses_sequence_parallel_size: 1
  meta_init: False
  groupgemm: False
  gradient_checkpointing: True
  save_hf: True
```

- `models.policy_trainer.fsdp_size`：设置 FSDP 并行组大小；默认包含所有可用 GPU。
- `models.policy_trainer.ulysses_sequence_parallel_size`：设为大于 1 时启用 Ulysses 序列并行。目前支持 **Qwen3-Dense** 和 **Qwen2.5-Dense**。
- `models.policy_trainer.meta_init`：是否使用meta init。模型权重仅在 rank 0 加载，并在初始化时广播到其他 rank。
- `models.policy_trainer.groupgemm`：将 Sequential MLP 替换为 GroupGEMM，目前仅支持 **Qwen3-Moe**。
- `models.policy_trainer.gradient_checkpointing`：启用gradient checkpointing，通过重新计算中间激活值来节省内存。
- `models.policy_trainer.save_hf`：若为 `True`，则在训练过程中保存 Hugging Face 格式的检查点。提供了一个 [离线合并脚本](https://github.com/alibaba/ChatLearn/blob/main/chatlearn/offline_ckpt_converter.py)，可将 FSDP 分布式权重合并为 Hugging Face 格式。

#### Megatron policy_trainer 配置

以下是 Megatron-Core 训练后端的特定配置：

```yaml
policy_trainer: 
  bf16: True
  seq_length: 2048
  tokenizer_type: 'HuggingFaceTokenizer'
  tokenizer_model: ${models.policy.load}
  tensor_model_parallel_size: 1
  pipeline_model_parallel_size: 1
  expert_tensor_parallel_size: null
  expert_model_parallel_size: 1
  virtual_pipeline_model_parallel_size: null
  decoder_first_pipeline_num_layers: null
  decoder_last_pipeline_num_layers: null
  moe_router_force_load_balancing: False
  # 训练配置
  load: your_megatron_model_path
  sequence_parallel: True
  use_distributed_optimizer: True
  recompute_granularity: null
  # 其他
  use_group_sequence_policy: False
```

- `models.policy_trainer.bf16`：启用 bfloat16 精度。若为 `False`，则使用 fp32。
- `models.policy_trainer.seq_length`：Megatron 训练的序列长度。若启用了 `packing`，此值将被忽略；否则必须与Rollout中seq_length一致。
- `models.policy_trainer.tokenizer_type`：Megatron 训练所用的 tokenizer 类型。大多数情况下推荐使用 `HuggingFaceTokenizer`。
- `models.policy_trainer.tokenizer_model`：Megatron 训练所用的 tokenizer 模型路径。
- `models.policy_trainer.tensor_model_parallel_size`：张量并行大小。
- `models.policy_trainer.pipeline_model_parallel_size`：流水线并行大小。
- `models.policy_trainer.expert_tensor_parallel_size`：专家张量并行大小。
- `models.policy_trainer.expert_model_parallel_size`：专家并行大小。
- `models.policy_trainer.virtual_pipeline_model_parallel_size`：虚拟流水线并行大小。当 `pipeline_model_parallel_size` 大于 1 时使用。
- `models.policy_trainer.decoder_first_pipeline_num_layers`：在流水线并行时第一个切分中decoder 层数。当模型层数无法被 `pipeline_model_parallel_size` 整除时使用。
- `models.policy_trainer.decoder_last_pipeline_num_layers`：在流水线并行时最后一个切分的 decoder 层数。同上。
- `models.policy_trainer.moe_router_force_load_balancing`：（吞吐测试时使用）若启用，则强制 MoE 路由器进行负载均衡，开启时影响强化学习训练收敛结果。
- `models.policy_trainer.load`：模型权重导入路径。
- `models.policy_trainer.sequence_parallel`：是否启用序列并行。当 `tensor_model_parallel_size` 大于 1 时启用。
- `models.policy_trainer.use_distributed_optimizer`：是否使用分布式优化器以减少内存占用。建议设为 `True`。
- `models.policy_trainer.recompute_granularity`：选择重计算粒度以节省内存。可选值为 `null`、`sel` 或 `full`。
- `models.policy_trainer.use_group_sequence_policy`：是否使用 [GSPO](https://qwenlm.github.io/blog/gspo/) 算法。

---

### ref_policy

`ref_policy` 与 `policy_trainer` 使用相同的后端，但需要独立配置。

#### 通用 ref_policy 配置

```yaml
ref_policy:
  free_gpu_memory:
    offload_weights: True
  generation_batch_size: 8
  gpu_per_process: 1
  num_gpu: ${models.policy_trainer.num_gpu}
  trainable: False
  load: ${models.policy_trainer.load}
  packing: ${models.policy_trainer.packing}
  max_token_in_packing: ${models.policy_trainer.max_token_in_packing}
```

- `models.ref_policy.free_gpu_memory.offload_weights`：启用权重卸载。在colocation场景中应设为 `True`（目前不支持非colocation场景）。
- `models.ref_policy.generation_batch_size`：计算 **ref_logprobs** 时单次batch_size大小。可与 `policy_trainer` 的不同。
- `models.ref_policy.trainable`：此模型应始终设为 **False**，因其仅用于推理。
- `models.ref_policy.packing`：与 `policy_trainer` 中功能相同，控制批次重组和打包。
- `models.ref_policy.max_token_in_packing`：当 `packing` 启用时用于重组，此值可与 `policy_trainer` 不同。
- `models.ref_policy.load`：基础模型检查点路径。通常应与 `policy_trainer` 一致。
- `models.ref_policy.gpu_per_process`：每个 Ray actor 分配的 GPU 数量。
- `models.ref_policy.num_gpu`：为此模型分配的总 GPU 数。

#### FSDP ref_policy 配置

FSDP 后端的自定义设置：

```yaml
ref_policy: 
  fsdp_size: ${models.policy_trainer.num_gpu}
  meta_init: False
  groupgemm: False
```

- `models.ref_policy.fsdp_size`、`meta_init`、`groupgemm`：与 `policy_trainer` 中的选项相同，但可独立设置以覆盖默认值。

#### Megatron ref_policy 配置

Megatron 后端的自定义设置：

```yaml
ref_policy:
  seq_length: ${models.policy_trainer.seq_length}
  tokenizer_type: 'HuggingFaceTokenizer'
  tokenizer_model: ${models.policy.load}
  bf16: True
  sequence_parallel: True
  tensor_model_parallel_size: ${models.policy_trainer.tensor_model_parallel_size}
  pipeline_model_parallel_size: ${models.policy_trainer.pipeline_model_parallel_size}
  expert_tensor_parallel_size: ${models.policy_trainer.expert_tensor_parallel_size}
  expert_model_parallel_size: ${models.policy_trainer.expert_model_parallel_size}
  decoder_first_pipeline_num_layers: ${models.policy_trainer.decoder_first_pipeline_num_layers}
  decoder_last_pipeline_num_layers: ${models.policy_trainer.decoder_last_pipeline_num_layers}
  moe_router_force_load_balancing: ${models.policy_trainer.moe_router_force_load_balancing}
  load: ${models.policy_trainer.load}
```

以上配置与 `policy_trainer` 相同，但可为 `ref_policy` 单独覆盖。然而，为确保强化学习目标函数计算的数值稳定性，建议保持两者一致，尤其是在 MoE 模型训练中。

---

### policy

SgLang 和 VLLM 使用相同的配置：

```yaml
policy:
  free_gpu_memory:
    offload_weights: True
  generation_batch_size: 256
  gpu_per_process: 1
  num_gpu: ${models.policy_trainer.num_gpu}
  tensor_model_parallel_size: 1
  trainable: False
  load: ${models.policy_trainer.load}
  num_inference_per_prompt: 32
  seq_length: 2048
  max_seq_len_to_capture: 2348
  temperature: 1.0
  top_p: 1.0
  eval_temperature: 0.6
  eval_top_p: 0.95
  eval_top_k: 20
  enable_thinking: False
  gpu_memory_utilization: 0.8
```

- `models.policy.free_gpu_memory.offload_weights`：若启用，则开启卸载模型权重。推荐用于colocation场景（目前不支持非colocation场景）。
- `models.policy.generation_batch_size`：用于vLLM 的 `max_num_seqs`。
- `models.policy.gpu_per_process`：每个 Ray actor 分配的 GPU 数量。
- `models.policy.num_gpu`：模型推理所用的总 GPU 数。
- `models.policy.tensor_model_parallel_size`：模型张量并行大小。
- `models.policy.trainable`：推理模型不可训练。
- `models.policy.load`：模型权重路径；GRPO 中应与 `policy_trainer.load` 一致。
- `models.policy.num_inference_per_prompt`：每个 prompt 生成的response数量。
- `models.policy.seq_length`：序列的最大长度（prompt length + response length）。
- `models.policy.max_seq_len_to_capture`：推理时捕获的最大序列长度。必须 ≥ `seq_length`。
- `models.policy.temperature`, `models.policy.top_p`：训练推理时的采样超参数。
- `models.policy.eval_temperature`, `models.policy.eval_top_p`, `models.policy.eval_top_k`：评估推理时的采样超参数。
- `models.policy.enable_thinking`：若启用，为 Qwen3 模型启用“思考模式”。
- `models.policy.gpu_memory_utilization`：推理引擎的GPU显存使用率。需谨慎设置，避免 OOM。

---

### reward

```yaml
models:
  reward:
    num_cpu: 2
    cpu_per_process: 1
    generation_batch_size: 256
```

- `models.reward.num_cpu`：rule-based reward actor 分配的总 CPU 数。
- `models.reward.cpu_per_process`：每个 reward actor 使用的 CPU 数量。
- `models.reward.generation_batch_size`： 单次 reward 前向计算的batch_size大小。