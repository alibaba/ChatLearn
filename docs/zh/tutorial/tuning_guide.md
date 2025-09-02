# 性能调优指南

本文档提供了一些建议，以最大化 ChatLearn 中的端到端训练效率。

## Megatron-Core

Megatron-Core 后端的性能主要取决于 `train_global_batch_size`（全局批次大小）、`train_micro_batch_size`（微批次大小）以及模型并行策略。此外，Megatron-Core 中的一些功能也会影响性能。以下是一些加速 Megatron-Core 后端训练的建议。更多详细信息，请参考 [Megatron-LM 文档](https://docs.nvidia.com/megatron-core/developer-guide/latest/index.html) 和 NeMo 框架的 [调优指南](https://docs.nvidia.com/nemo-framework/user-guide/latest/performance/performance-guide.html#performance-tuning-guide)。

### 模型并行

由于 `train_global_batch_size` 通常由算法本身的需求决定，本章节旨在提供如何调整模型并行策略的建议，以便在给定的 `train_global_batch_size` 下实现良好的性能。

大多数模型并行技术（TP、PP、ETP、EP、CP）旨在将模型的内存消耗分布到多个 GPU 上，但会带来额外的通信开销。因此，选择合适的并行配置以避免内存溢出（OOM）并获得高吞吐量至关重要。

+ **上下文并行 (CP)**：用于长上下文训练。当序列长度 ≥ 8192 时，建议设置为 2 或更大。
+ **张量并行 (TP)**：将 GEMM 计算分配到多个 GPU 上。不要超过单节点 GPU 数量（通常为 8）。
+ **流水线并行 (PP)**：将模型拆分为多个阶段，并以流水线方式执行前向和反向传播。当 `train_global_batch_size` 较大时，可以在较低开销下显著减少内存消耗。建议 PP 大小小于 `num_microbatches`（微批次数量）。
+ **专家并行 (EP)**：用于 MoE（混合专家）层。
+ **专家张量并行 (ETP)**：MoE 层的张量并行。根据 NVIDIA 的建议，EP 通常能获得更好的性能，而 ETP 可以节省更多内存。如果不会发生 OOM，应优先用 EP 替代 ETP。

作为起点，我们建议搜索每 GPU 约 40 亿参数的模型并行配置，即经验上，对于稠密模型，`n_params / (TP * PP)` 或对于 MoE 模型，`n_params / (ETP * EP)` 应接近 40 亿。如果 `train_global_batch_size` 相对较大，建议 PP 至少为 2，并使用虚拟流水线并行（VPP）以实现更好的重叠（可尝试 vpp_size=1 或 2）；否则，TP/CP 可能表现更佳。

如果未开启动态批处理（dynamic batching），在搜索最佳模型并行设置时，可将 `train_micro_batch_size` 设置为 1，在确定设置后再进行放大。

### 动态批处理

动态批处理旨在自动将未填充的样本合并到若干个打包序列中，通过消除填充（pad）token 可显著提升性能。默认情况下此功能是开启的，如果需要调试时禁用，可以设置 `models.policy_trainer.packing=false`。此外，应设置合适的 `models.policy_trainer.max_token_in_packing` 以避免 OOM（同时不要设置过小以免影响效率）。

### 通信重叠与算子融合

通信重叠（Overlapping）和算子融合（OP Fusion）也有助于提升训练效率。然而，这些功能在 ChatLearn 中默认是关闭的，直到模型收敛性得到确认。如果希望启用它们，请参考 Megatron-LM 的文档，并将相关参数添加到 `chatlearn.configs.megatron_config.MegatronConfig` 中。随后，ChatLearn 会自动将这些参数传递给 Megatron 以启用相应功能。

### 激活值重计算

激活值重计算（Activation Recomputation）是一种在用户不追求训练效率、而希望使用最少的机器资源来训练模型时，用于减少内存使用的策略。该功能默认是关闭的。设置 `models.policy_trainer.recompute_granularity='selective'` 可以节省内存且对训练效率影响较小；设置为 `'full'` 可以最大程度地节省内存（但速度较慢）。