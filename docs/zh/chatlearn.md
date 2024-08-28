# ChatLearn: 灵活、易用、高效的大规模 Alignmant 训练框架

## 概述

ChatGPT 是由 OpenAI 开发的基于大型语言模型 (Large Language Model, LLM) 的聊天机器人，以其令人惊叹的对话能力而迅速火爆并被广泛采用。ChatGPT 成功背后得益于大型语言模型生成领域的新训练范式：RLHF (Reinforcement Learning from Human Feedback)，即以强化学习方式依据人类反馈优化语言模型。

![RLHF Flow](../images/rlhf.png)

不同于传统的深度学习训练，只涉及到单个模型的迭代和优化，以 RLHF 为代表的训练范式涉及到多个大模型的计算和数据交互，这对于构建构建一个易用、高效的训练系统提出了诸多的挑战。
1. **编程接口**: 如何设计一个通用且灵活的编程接口，让用户能专注于单个模型的建模，同时，又可以灵活地控制模型间的交互。
2. **分布式加速引擎**: 随着模型规模的增大，用户会选择一些分布式训练和推理加速的 backend，比如 Megatron-LM、DeepSpeed、vLLM 等，如何结合这些加速 backend 来实现高效的多模型计算框架。
3. **并行策略**: 多个模型可能各有各的计算特点，比如仅推理的模型和训练的模型在显存和计算上的特性都不同，每个模型最佳的并行策略也可能不同。因此，框架应该允许不同的模型配置不同的并行策略以发挥整体的最佳性能。
4. **资源分配**: 如何灵活地给多个模型分配资源来实现高效的并发调度和执行。

为了解决上述问题，我们提出了一个新的 Alignment 训练框架 ChatLearn。ChatLearn 通过对模型计算逻辑的抽象，解耦了模型和计算 backend、分布式策略的绑定，提供灵活的资源调度机制，可以支持灵活的资源分配和并行调度策略。ChatLearn 的优点总结如下:
1. **易用的编程接口**: ChatLearn提供通用的编程抽象，用户只需要封装几个函数即可完成模型构造。用户只需要专注于单模型的编程，系统负责资源调度、数据流传输、控制流传输、分布式执行等。
2. **高可扩展的训练方式**: ChatLearn 提供 RLHF、DPO、OnlineDPO、GRPO 等 Alignment 训练，同时也支持用户自定义 model 的执行 flow，使定制化训练流程变得非常便捷。
3. **多种分布式加速引擎**: 用户可以使用不同的计算 backend 进行模型建模，如 Megatron-LM、DeepSpeed、vLLM 等。用户也可以组合使用不同的 backend，如用 Megatron-LM 来进行加速训练，用 vLLM 来加速推理。
4. **灵活的并行策略和资源分配**: ChatLearn 支持不同模型配置不同的并行策略，可以结合各模型计算、显存、通信的特点来制定不同的并行策略。同时 ChatLearn 支持灵活的资源调度机制，支持各模型的资源独占或复用，通过系统调度策略支持高效的串行/并行执行和高效的显存共享。
5. **高性能**: 相较于当前的 SOTA 系统，ChatLearn 在 7B+7B (Policy+Reward) 规模性能提升52%，70B+70B 规模性能提升 137%。同时，ChatLearn 支持更大规模的 Alignment 训练，例如：300B+300B。

## 技术架构

![arch](../images/arch.png)

**API:** ChatLearn提供了RLHF、DPO、OnlineDPO、GRPO 等 Alignment 训练，同时也支持用户自定义 model 的执行 flow，来实现自定义的训练流程。同时ChatLearn提供Module的抽象，用户通过继承MegatronModule、DeepSpeedModule、VLLMModule 完成对不同计算backend的封装。ChatLearn 通过 yaml 文件的形式为 Alignment 训练，以及不同的模型配置不同的超参数、并行策略等，来实现灵活的模型和并行策略配置。

**Scheduler:** ChatLearn 提出了 DistActor 的抽象来支持模型的分布式训练或推理。DistActor 继承了 Ray actor 的状态管理和 worker 间的隔离性，同时突破了 Ray actor 不能跨机的限制。通过 DistActor，ChatLearn 可以支持任意规模的模型推理和训练。同时，ChatLearn Scheduler 通过划分集群 Resource Group 和调度策略，实现硬件感知的亲和性调度。ChatLearn 也支持灵活的资源分配，支持模型间的资源复用、独占或部分复用等策略，在给定资源数的情况下，实现训练效率的最大化。

**Executor:** ChatLearn Executor 将 Alignment 训练流程划分为三个主要的模块，`Environment`、 `Trainer`和 `Evaluator`。`Environment` 负责推理模块模型和数据的并发执行和管理，`Trainer` 负责相应的训练模块，`Evaluator` 负责模型效果评估。Executor 还负责数据传输、参数同步。

**Backend:** 得益于 ChatLearn 良好的编程接口抽象，用户通过简单的封装即可接入各种不同 backend 进行计算优化和算法优化。

**Optimization:** ChatLearn 也支持各种计算、显存、通信优化，通过各种并行策略组合来加速训练，通过 paged attention 和 continuous batching 等来加速推理，通过 EMS(Efficient Memory Sharing) 技术来高效复用显存，减少总资源需求，通过分组广播技术来支持 Training 和 Inference 模型间高效参数同步，等等。


## 快速开始

请参考 [文档](https://chatlearn.readthedocs.io/zh-cn/latest/) 快速开始.

1. [环境和代码准备](installation.md)
2. [基于 Llama/Llama2 模型的端到端训练教程](tutorial/tutorial_llama2.md)


## 性能评估

我们比较了不同参数量规模模型的 RLHF 训练吞吐量，我们采取 N+N 的模型配置，即 Policy 模型和 Reward 模型采用相同大小的参数量。我们和 DeepSpeed-Chat、OpenRLHF 对比了 7B 和 70B 的模型配置，在 8 GPUs 7B+7B 规模，有 115% 的加速，在 32 GPUs 70B+70B 规模，有 208% 的加速。规模越大，加速效果越明显。同时ChatLearn还能支持更大规模的 Alignment 训练，例如：300B+300B 规模。

![compare perf](../images/perf.png)

注：DeepSpeed-Chat和OpenRLHF性能已经优化过。

## Roadmap

ChatLearn 接下来会支持以下特性：
- [ ] 支持Megatron-Core格式模型；
- [ ] 支持MoE模型Alignment训练；
- [ ] 接入 DeepSpeed 作为训练 backend；
- [ ] 支持更多的模型；
- [ ] 性能优化；
- [ ] 支持更多的 Alignment 算法；

<br><br>
我们欢迎社区小伙伴参与进来合作开发。

## Reference

1. Megatron-LM: https://github.com/NVIDIA/Megatron-LM
2. DeepSpeed-Chat: https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat
3. OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
