# Performance Tuning Guide

This document provides suggestions to maximize e2e training efficiency in ChatLearn.

## Megatron-Core

The performance of Megatron-Core backend is mainly dependent on the `train_global_batch_size`, `train_micro_batch_size` and model parallelism.
Besides, some features in Megatron-Core also affect the performance. Here are some suggestions to accelerate training with Megatron-Core backend.
For more details, please refer to the [Documentation](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html) of Megatron-LM and
[Tuning Guide](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html#performance-tuning-guide) of NeMo Framework.

### Model Parallelism

As `train_global_batch_size` is determined by algorithm-specific considerations, this chapter aims to provide
some insights on how to tune model parallelism to reach good performance with the given `train_global_batch_size`.

Most of Model Parallelism (TP, PP, ETP, EP, CP) aims to distribute the memory consumption of the model across multiple GPUs with the extra 
communication costs. Therefore, it is important to choose a proper parallelism config to avoid OOM and get high throughput.

+ Context Parallel (CP): Used for long-context training. Recommended to set 2 or larger when sequence length >= 8192.
+ Tensor Parallel(TP): Divide GEMM computation to multiple GPUs. Do not exceed GPU-per-node (8).
+ Pipeline Parallel (PP): Split the model to multiple stages and run forward-backward in pipeline. Can significantly 
reduce memory consumption with low cost when `train_global_batch_size` is large. Suggest to be smaller than `num_microbatches`.
+ Expert Parallel (EP): Used for MoE Layer.
+ Expert Tensor Parallel (ETP): Tensor Parallel for MoE Layer. As NVIDIA suggest, EP can reach better performance while ETP can save
more memory. Always replace ETP with EP if no OOM occurs.

As a starting point, we recommend to search a proper model parallel setting with 4B parameters per GPU, i.e., emprically,
`n_params / (TP * PP)` (for dense model) or `n_params / (ETP * EP)` (for MoE model) should be around 4B. If `train_global_batch_size`
is relatively large, PP is suggested to be at least 2 and VPP is used for better overlapping (try vpp_size=1 or 2), otherwise, TP/CP may have
better performance.

If dynamic batching is not open, for `train_micro_batch_size`, you can set to 1 when searching for the best model parallel settings, and scale up when the settings are determined.

### Dynamic batching

Dynamic batching is designed to automatically merge the unpadded samples of a global batch into several packed sequence, and can obtain significant
performance improvement due to the removal of pad tokens. By default this feature is enabled, if you want to disable for debugging, you can set `models.policy_trainer.packing=false`.
Besides, set a proper `models.policy_trainer.max_token_in_packing` to avoid OOM (and do not be too small for efficiency).

### Overlapping and Fusion

Communication Overlapping and OP Fusion are also benefit for training efficiency. However, these features are disabled by default in ChatLearn until the convergence is confirmed. If you want to enable them, please refer to the docs of Megatron-LM and add the arguments in `chatlearn.configs.megatron_config.MegatronConfig`. Then ChatLearn will automatically pass them to the Megatron to enable the features.

### Activation Recomputation

Activation recomputation is a choice to reduce memory usage when user does not care the training efficiency, but want to use the minimal machines to train the model. It is disabled by default.
Set `models.policy_trainer.recompute_granularity='selective'` to save memory but do not harm much to training efficiency, or `models.policy_trainer.recompute_granularity='full'` to save memory as much as possible (slow).