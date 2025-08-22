# Config Explanation
ChatLearn's configuration comprises two main components:
- `runtime_args`: Primary training configurations for the framework.
- `models`: Configuration settings for each individual model.

Configuration templates are provided in ChatLearn/template.

## runtime_args
```yaml
runtime_args:
  # setup config
  train_backend: fsdp
  rollout_backend: vllm
  exp_name: grpo_fsdp
  colocation: [policy,policy_trainer,ref_policy]
  # path config
  output_dir: your_output_dir
  data_path: your_data_path
  eval_data_path: your_eval_data_path
  data_checkpoint_path: ${runtime_args.output_dir}/data_checkpoint_path/
  # config for training
  num_episode: 200
  sample_per_episode: 512
  train_global_batch_size: 512
  save_episode_interval: 200
  # config for data
  data_shuffle: True
  data_rerank: True
  # config for eval
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
- `runtime_args.train_backend`: Training backend to use; supports **fsdp** or **megatron**
- `runtime_args.rollout_backend`: Rollout backend to use; choose between **vllm** or **sglang**
- `runtime_args.exp_name`: Experiment name, used for logging metrics.
- `runtime_args.colocation`: Models listed here will be colocated on the GPU, and will execute sequentially.
- `runtime_args.output_dir`: Directory to save all intermediate training results.
- `runtime_args.data_path`: Training data path. Ensure data files in this directory are compatible with the data reading code.
- `runtime_args.eval_data_path`: Evaluation data path. Ensure data files in this directory are compatible with the data reading code.
- `runtime_args.data_checkpoint_path`: Path for saving data checkpoints. The default is **data_checkpoint_path/** under your `runtime_args.output_dir`
- `runtime_args.num_episode`: Total number of training episodes. Each episode involves several weight updates.
- `runtime_args.sample_per_episode`: Number of training samples per episode(sample_per_episode=prompt_per_episode*num_inference_per_prompt).
- `runtime_args.train_global_batch_size`: `runtime_args.sample_per_episode` will be divided into multiple global batches of this size. Each batch is used in one training round (global across actors).
- `runtime_args.save_episode_interval`: Interval for saving intermediate checkpoints.
- `runtime_args.eval_episode_interval`: Interval between evaluation rounds.
- `runtime_args.data_shuffle`: If enabled, dataset samples will be shuffled, ignoring the original order.
- `runtime_args.data_rerank`: If enabled, multiple replicas of the same data sample will not be assigned to the same rollout actor.
- `runtime_args.enable_eval_before_training`: Whether to do evaluation before training.
- `runtime_args.log_args_dict`: Logging configuration. Ensure you are logged in to Weights & Biases when `runtime_args.enable_wandb=True`.

## models
For the GRPO algorithm, ChatLearn uses four models: `policy_trainer`, `ref_policy`, `policy`, `reward`. 
*Note: policy_trainer and ref_policy share the same training backend.*
### policy_trainer
#### Common config
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
- `models.policy_trainer.free_gpu_memory.*`: Controls GPU memory offloading; set all to True for colocation scenarios (non-colocation is not supported yet).
- `models.policy_trainer.optimizer.lr`: Learning rate.
- `models.policy_trainer.optimizer.clip_grad`: Gradient clipping rate.
- `models.policy_trainer.trainable`: Enable training, this should always be True for trainer
- `models.policy_trainer.generation_batch_size`: Batch size for a single model forward pass (used to compute **old_logprobs**).
- `models.policy_trainer.train_micro_batch_size`: Local batch size per model for forward/backward pass; used for gradient accumulation.
- `models.policy_trainer.packing`: If enabled, samples are regrouped into batches with fewer total tokens then `models.policy_trainer.max_token_in_packing` in each batch. After regrouping, each batch will be packed into single sequence. If enabled, `models.policy_trainer.generation_batch_size` and `models.policy_trainer.train_micro_batch_size` will be ignored.
- `models.policy_trainer.max_token_in_packing`: Used for regroup when `models.policy_trainer.packing` is enabled
- `models.policy_trainer.load`: Path to load base checkpoint model.
- `models.policy_trainer.pos_clip_ratio`, `models.policy_trainer.neg_clip_ratio`: GRPO algorithm coefficients.
- `models.policy_trainer.entropy_coef`: Set above 0.0 to enable entropy loss in backward pass.
- `models.policy_trainer.kl_coef`: Set above 0.0 to enable KL loss in backward pass.
- `models.policy_trainer.gpu_per_process`: GPUs assigned to each Ray actor.
- `models.policy_trainer.num_gpu`: Total GPUs for training (on multinode, this is across all nodes).

#### FSDP policy_trainer config
The following are specific configuration options for the FSDP training backend:
```yaml
policy_trainer: 
  fsdp_size: ${models.policy_trainer.num_gpu}
  ulysses_sequence_parallel_size: 1
  meta_init: False
  groupgemm: False
  gradient_checkpointing: True
  save_hf: True
```
- `models.policy_trainer.fsdp_size`: Sets the FSDP parallel group size; by default, this includes all available GPUs.
- `models.policy_trainer.ulysses_sequence_parallel_size`: Enables Ulysses sequence parallelism when set greater than 1. Currently support **Qwen3-Dense** and **Qwen2.5-Dense**.
- `models.policy_trainer.meta_init`: Enables meta initialization for the FSDP wrapper. Model weights are loaded only on rank 0 and broadcasted to other ranks during setup.
- `models.policy_trainer.groupgemm`: Replace Sequential MLP with GroupGEMM, currently only support **Qwen3-Moe**.
- `models.policy_trainer.gradient_checkpointing`: Enables recomputation of intermediate activations during training to save memory (gradient checkpointing).
- `models.policy_trainer.save_hf`: If True, saves Hugging Face format checkpoints during training. An [offline merge script](https://github.com/alibaba/ChatLearn/blob/main/chatlearn/offline_ckpt_converter.py) is provided to merge FSDP distributed checkpoints into a Hugging Face checkpoint.

#### Megatron policy_trainer config
The following are specific configuration options for the Megatron-Core training backend:
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
  # train config
  load: your_megatron_model_path
  sequence_parallel: True
  use_distributed_optimizer: True
  recompute_granularity: null
  # other 
  use_group_sequence_policy: False
```

- `models.policy_trainer.bf16`: Enables bfloat16 precision. If set to False, fp32 will be used.
- `models.policy_trainer.seq_length`: Sequence length for Megatron Training. If `models.policy_trainer.packing` is enabled, the value will be ignored.
Otherwise, the value must be equal to the value used for data generation.
- `models.policy_trainer.tokenizer_type`: Tokenizer type for Megatron Training. For most cases, HuggingFaceTokenizer is recommended.
- `models.policy_trainer.tokenizer_model`: Path to the tokenizer model for Megatron training.
- `models.policy_trainer.tensor_model_parallel_size`: Tensor model parallel world size.
- `models.policy_trainer.pipeline_model_parallel_size`: Pipeline model parallel world size.
- `models.policy_trainer.expert_tensor_parallel_size`: Expert tensor model parallel world size.
- `models.policy_trainer.expert_model_parallel_size`:Expert model parallel world size.
- `models.policy_trainer.virtual_pipeline_model_parallel_size`: Virtual pipeline model parallel world size. Used when `pipeline_model_parallel_size` larger than 1.
- `models.policy_trainer.decoder_first_pipeline_num_layers`: Number of decoder layers of the first pipeline stage. Used when num_layers of the model cannot be divided by `pipeline_model_parallel_size`.
- `models.policy_trainer.decoder_last_pipeline_num_layers`: Number of decoder layers of the last pipeline stage. Used when num_layers of the model cannot be divided by `pipeline_model_parallel_size`.
- `models.policy_trainer.moe_router_force_load_balancing`: (Benchmarking) Forces load balancing for MoE routers if enabled.
- `models.policy_trainer.load`: Path to the model checkpoint.
- `models.policy_trainer.sequence_parallel`: Whether to use sequence parallelism. Valid when `tensor_model_parallel_size` larger than 1.
- `models.policy_trainer.use_distributed_optimizer`: Whether to use distributed optimizer to reduce memory consumption. Recommended to set True.
- `models.policy_trainer.recompute_granularity`: Select recompute granularity to save memory usage. Should be null, `sel` or `full`. 
- `models.policy_trainer.use_group_sequence_policy`: Whether to use [GSPO](https://qwenlm.github.io/blog/gspo/).


### ref_policy
ref_policy uses the same backend as policy_trainer, but can be customized separately.
### common ref_policy config
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
- `models.ref_policy.free_gpu_memory.offload_weights`: Enables offloading of weights. This should be set to True in colocation scenarios (non-colocation is not supported yet).
- `models.ref_policy.generation_batch_size`: Batch size used for each forward pass when computing **ref_logprobs**. It can differ from `models.policy_trainer.generation_batch_size`
- `models.ref_policy.trainable`: Should always be **False** for the this model, as it only performs inference.
- `models.ref_policy.packing`:  Same functionality as in `models.policy_trainer.packing`—controls batch regrouping and packing.
- `models.ref_policy.max_token_in_packing`: Used for regroup when `models.ref_policy.packing` is enabled. This value can differ from `models.policy_trainer.packing`
- `models.ref_policy.load`:Path to the base model checkpoint. This should match `models.policy_trainer.load` unless a different checkpoint is needed.
- `models.ref_policy.gpu_per_process`: Number of GPUs to assign to each Ray actor.
- `models.ref_policy.num_gpu`: Total number of GPUs allocated for this model. In multinode training, this represents the total across all nodes.

#### FSDP ref_policy config
Custom settings for the FSDP training backend.
```yaml
ref_policy: 
  fsdp_size: ${models.policy_trainer.num_gpu}
  meta_init: False
  groupgemm: False
```
- `models.ref_policy.fsdp_size`, `models.ref_policy.meta_init`, and `models.ref_policy.groupgemm`: These mirror the options in `models.policy_trainer`, but can be set independently to override the defaults.

#### Megatron ref_policy config
Custom settings for the Megatron training backend.

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

All the above configurations are the same as policy trainer, but can be overridden for reference policy model. However, to improve the numerical stability, we recommend to keep two models consistent, especially in MoE training.


### policy
SgLang and Vllm share same configuration
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
- `models.policy.free_gpu_memory.offload_weights`: If enabled, model weights are offloaded. Recommended for colocation scenarios. Non-colocation is not supported yet.
- `models.policy.generation_batch_size`: Sets **max_num_seqs** for VLLM
- `models.policy.gpu_per_process`: Number of GPUs assigned per Ray actor.
- `models.policy.num_gpu`: Total GPUs allotted for policy inference. In multinode setups, sums across all nodes.
- `models.policy.tensor_model_parallel_size`: Specifies tensor parallel size for model parallelism.
- `models.policy.trainable`: Policy model is non-trainable.
- `models.policy.load`: Path to model checkpoint; should match `models.policy_trainer.load` for GRPO.
- `models.policy.num_inference_per_prompt`: Number of responses generated per prompt.
- `models.policy.seq_length`: Maximum response sequence length(prompt length + response length).
- `models.policy.max_seq_len_to_capture`: Max sequence length captured during inference. Must be ≥ `models.policy.seq_length`.
- `models.policy.temperature`, `models.policy.top_p`: Sampling hyperparameters for training rollouts.
- `models.policy.eval_temperature`, `models.policy.eval_top_p`, `models.policy.eval_top_k`: Sampling hyperparameters for evaluation rollouts.
- `models.policy.enable_thinking`:  Enables "thinking mode" for Qwen3 models, if applicable.
- `models.policy.gpu_memory_utilization`: Target GPU memory utilization for the rollout engine. Use with caution in colocation mode to avoid OOM.

### reward
```yaml
models:
  reward:
    num_cpu: 2
    cpu_per_process: 1
    generation_batch_size: 256
```
- `models.reward.num_cpu`: Total CPUs to allocate for the rule-based (CPU-based) reward actors.
- `models.reward.cpu_per_process`: Number of CPUs used by each individual reward actor.
- `models.reward.generation_batch_size`: Batch size for a single reward forward computation.