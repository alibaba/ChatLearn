
[![docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://chatlearn.readthedocs.io/zh-cn/latest/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibaba/ChatLearn/blob/main/LICENSE)

<p align="center">
  <picture>
    <img alt="ChatLearn" src="docs/images/logo.jpg" width=30%>
  </picture>
</p>

<h3 align="center">
灵活、易用、高效的大语言模型（LLMs）强化学习训练框架
</h3>
<p align="center">
        <a href="README.md">English</a>&nbsp |  &nbsp中文&nbsp
</p>

---

*最新进展* 🔥
- [2025/7] 提供基于[Mcore](scripts/train_mcore_vllm_deepseek_v3_671b_grpo.sh)的DeepSeek-V3-671B强化学习训练示例!🔥
- [2025/7] 提供基于[Mcore](scripts/train_mcore_vllm_qwen3_235b_grpo.sh)和[FSDP2](scripts/train_fsdp_vllm_qwen3_235b_a22b_grpo.sh)的Qwen3-235B-A22B强化学习训练示例!🔥
- [2025/7] 训练支持FSDP2框架！提供sequence packing，sequence parallelism，group GEMM支持实现高效易用的强化学习训练!🔥
- [2025/5] 训练支持Mcore框架！基于Mcore和vLLM，我们提供了Qwen3模型的端到端GRPO训练[教学](docs/en/tutorial/tutorial_grpo_mcore.md)!
- [2025/5] 训练支持FSDP框架！基于FSDP和vLLM，我们提供了Qwen3模型的端到端GRPO训练[教学](docs/en/tutorial/tutorial_grpo_fsdp.md)!
- [2024/8] 正式开源 ChatLearn，更多介绍请参考我们的 [文档](docs/zh/chatlearn.md)。

---

ChatLearn 是阿里云PAI团队开发的大规模LLMs强化学习训练框架。ChatLearn 通过对模型计算逻辑的抽象，解耦了模型和计算 backend、分布式策略的绑定，提供灵活的资源调度机制，可以支持灵活的资源分配和并行调度策略。

![RLHF Flow](docs/images/rlhf.png)

ChatLearn的特点如下:
1. 🚀**易用的编程接口**: ChatLearn提供通用的编程抽象，用户只需要封装几个函数即可完成模型构造。用户只需要专注于单模型的编程，系统负责资源调度、数据流传输、控制流传输、分布式执行等。
2. 🔧**高可扩展的训练方式**: ChatLearn 提供 RLHF、DPO、OnlineDPO、GRPO 等 强化学习训练，同时也支持用户自定义 model 的执行 flow，使定制化训练流程变得非常便捷。
3. 🔄**多种分布式加速引擎**: 用户可以使用不同的计算 backend 进行模型建模，如 Megatron-LM、DeepSpeed、vLLM 等。用户也可以组合使用不同的 backend，如用 Megatron-LM 来进行加速训练，用 vLLM 来加速推理。
4. 🎯**灵活的并行策略和资源分配**: ChatLearn 支持不同模型配置不同的并行策略，可以结合各模型计算、显存、通信的特点来制定不同的并行策略。同时 ChatLearn 支持灵活的资源调度机制，支持各模型的资源独占或复用，通过系统调度策略支持高效的串行/并行执行和高效的显存共享。
5. ⚡**高性能**: 相较于当前的 SOTA 系统，ChatLearn 在 7B+7B (Policy+Reward) 规模性能提升52%，70B+70B 规模性能提升 137%。同时，ChatLearn 支持更大规模的 Alignment 训练，例如：300B+300B。

# 快速开始

请参考 [文档](https://chatlearn.readthedocs.io/zh-cn/latest/) 快速开始.

1. [环境和代码准备](docs/zh/installation.md)
2. [基于 FSDP + vLLM的Qwen3模型端到端GRPO训练流程](docs/zh/tutorial/tutorial_grpo_fsdp.md)
3. [基于 LLaMA/LLaMA2 模型的端到端训练教程](docs/zh/tutorial/tutorial_llama2.md)


# 性能评估

我们比较了不同参数量规模模型的 RLHF 训练吞吐量，我们采取 N+N 的模型配置，即 Policy 模型和 Reward 模型采用相同大小的参数量。我们和 DeepSpeed-Chat、OpenRLHF 对比了 7B 和 70B 的模型配置，在 8 GPUs 7B+7B 规模，有 115% 的加速，在 32 GPUs 70B+70B 规模，有 208% 的加速。规模越大，加速效果越明显。同时ChatLearn还能支持更大规模的 Alignment 训练，例如：300B+300B 规模。


![Compare Performance](docs/images/perf.png)

注：DeepSpeed-Chat和OpenRLHF性能已经优化过。

# 功能列表

- 支持 RLHF、DPO、OnlineDPO、GRPO 以及用户自定义的RL训练；
- 支持 Megatron-LM，FSDP 作为训练的 backend，支持 vLLM 作为推理的 backend；
- 支持 各模型独立配置并行策略，并支持模型间高效参数同步，自动进行并行策略转换；
- 支持 EMS(Efficient Memory Sharing) 功能，支持模型间显存高效共享；
- 支持模型的资源类型：GPU、CPU，例如定义纯 CPU 的 Math Reward 模型；
- 支持 Megatron-Core 格式模型；

# Roadmap

ChatLearn 接下来会支持以下特性：
- [X] 简化参数配置
- [X] 提供MoE模型强化学习训练的教程
- [ ] 支持更多的模型
- [ ] 性能优化
- [ ] 支持更多的强化学习算法

<br><br>
我们欢迎社区小伙伴参与进来合作开发，也欢迎加入钉钉群：98090003312 参与讨论。我们在持续招聘中，欢迎联系我们或者投递简历到[email](mailto:wanglin.zj@alibaba-inc.com)。
