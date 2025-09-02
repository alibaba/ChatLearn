# ChatLearn: 灵活、易用、高效的大语言模型（LLMs）强化学习训练框架

## 概述

ChatLearn 是阿里云PAI团队开发的大规模LLMs强化学习训练框架。ChatLearn 通过对模型计算逻辑的抽象，解耦了模型和计算 backend、分布式策略的绑定，提供灵活的资源调度机制，可以支持灵活的资源分配和并行调度策略。

![RLHF Flow](../images/rlhf.png)

ChatLearn的特点如下:
1. 🚀**易用的编程接口**: ChatLearn提供通用的编程抽象，用户只需要封装几个函数即可完成模型构造。用户只需要专注于单模型的编程，系统负责资源调度、数据流传输、控制流传输、分布式执行等。
2. 🔧**高可扩展的训练方式**: ChatLearn 支持用户自定义模型执行流，使定制化训练流程更加灵活便捷。
3. 🔄**多种分布式加速引擎**: ChatLearn支持业界SOTA训练（FSDP2，Megatron）和推理引擎（vLLM， SGLang），实现卓越的训练吞吐能力
4. 🎯**灵活的并行策略和资源分配**: ChatLearn 支持不同模型配置不同的并行策略，可以结合各模型计算、显存、通信的特点来制定不同的并行策略。同时 ChatLearn 支持灵活的资源调度机制，支持各模型的资源独占或复用，通过系统调度策略支持高效的串行/并行执行和高效的显存共享。
5. ⚡**高性能**: 相较于当前的 SOTA 系统，ChatLearn 在 7B+7B (Policy+Reward) 规模性能提升52%，70B+70B 规模性能提升 137%。同时，ChatLearn 支持600B+规模的强化学习训练。


## 快速开始

请参考 [文档](https://chatlearn.readthedocs.io/zh-cn/latest/) 快速开始.

1. [环境和代码准备](installation.md)
2. [基于 FSDP + vLLM的Qwen3模型端到端GRPO训练流程](tutorial/tutorial_grpo_fsdp.md)
3. [基于 Megatron + vLLM的Qwen3模型端到端GRPO训练流程](tutorial/tutorial_grpo_mcore.md)

## 功能列表

- 支持[Megatron](https://github.com/alibaba/ChatLearn/blob/main/scripts/train_mcore_vllm_qwen3_8b_grpo.sh)、[FSDP](https://github.com/alibaba/ChatLearn/blob/main/scripts/train_fsdp_vllm_qwen3_8b_grpo.sh)训练引擎
- 支持vLLM、SGLang推理引擎，通过`runtime_args.rollout_engine`参数进行控制
- 支持GRPO、[GSPO](https://github.com/alibaba/ChatLearn/blob/main/scripts/train_mcore_vllm_qwen3_30b_gspo.sh)等强化学习算法
- 支持使用wandb、tensorboard监控实验
- 支持[sequence packing](https://github.com/alibaba/ChatLearn/blob/main/scripts/train_fsdp_vllm_qwen3_8b_grpo.sh)、ulysses sequence parellel、[Group GEMM](https://github.com/alibaba/ChatLearn/blob/main/scripts/train_fsdp_vllm_qwen3_30b_a3b_grpo.sh)等训练加速技术

## 性能评估

我们比较了不同参数量规模模型的 RLHF 训练吞吐量，我们采取 N+N 的模型配置，即 Policy 模型和 Reward 模型采用相同大小的参数量。我们和 DeepSpeed-Chat、OpenRLHF 对比了 7B 和 70B 的模型配置，在 8 GPUs 7B+7B 规模，有 115% 的加速，在 32 GPUs 70B+70B 规模，有 208% 的加速。规模越大，加速效果越明显。同时ChatLearn还能支持更大规模的强化学习训练，例如：600B 规模。

![compare perf](../images/perf.png)

注：DeepSpeed-Chat和OpenRLHF性能已经优化过。

## Roadmap

ChatLearn 接下来会支持以下特性：
- [X] 简化参数配置
- [X] 提供MoE模型强化学习训练的教程
- [ ] 支持更多的模型
- [ ] 性能优化
- [ ] 支持更多的强化学习算法
