
[![docs](https://img.shields.io/badge/docs-latest-brightgreen.svg)](https://chatlearn.readthedocs.io/en/latest/)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/alibaba/ChatLearn/blob/main/LICENSE)

<p align="center">
  <picture>
    <img alt="ChatLearn" src="docs/images/logo.jpg" width=30%>
  </picture>
</p>

<h3 align="center">
A flexible and efficient reinforcement learning framework for large language models(LLMs).  
</h3>

<p align="center">
        &nbspEnglish&nbsp |  <a href="README_CN.md"> 中文 </a>&nbsp
</p>


---

*Latest News* 🔥
- [2025/7] We give a reinforcement learning training example for DeepSeek-V3-671B based on [Mcore](scripts/train_mcore_vllm_deepseek_v3_671b_grpo.sh)! 🔥
- [2025/7] We give reinforcement learning training examples for Qwen3-235B-A22B based on [Mcore](scripts/train_mcore_vllm_qwen3_235b_grpo.sh) and [FSDP2](scripts/train_fsdp_vllm_qwen3_235b_a22b_grpo.sh)! 🔥
- [2025/7] Training now supports the FSDP2 framework! We support sequence packing, sequence parallelism, and group GEMM for efficient and user-friendly reinforcement learning training! 🔥
- [2025/5] We support Mcore frameworks for training! By using Mcore and vLLM, we give a [tutorial](docs/en/tutorial/tutorial_grpo_mcore.md) about end-2-end GRPO training for Qwen3!
- [2025/5] We support FSDP frameworks for training! By using FSDP and vLLM, we give a [tutorial](docs/en/tutorial/tutorial_grpo_fsdp.md) about end-2-end GRPO training for Qwen3!
- [2024/8] We officially released ChatLearn! Check out our [documentation](docs/en/chatlearn.md).

---

ChatLearn is a large-scale reinforcement learning training framework for LLMs developed by the Alibaba Cloud PAI platform.

![RLHF Flow](docs/images/rlhf.png)

Chatlearn has the following advantages:
1. 🚀**User-friendly programming interface**: Users can focus on programming individual models by wrapping a few functions, while the system takes care of resource scheduling, data and control flow transmission, and distributed execution.
2. 🔧**Highly Scalable Training Methodology**: ChatLearn offers RL training such as RLHF, DPO, OnlineDPO and GRPO, while also supporting user-defined execution flows for models, enabling a highly convenient and customizable training process.
3. 🔄**Diverse Distributed Acceleration Engines**: Users can leverage various computational backends for model construction, such as Megatron-LM, DeepSpeed, vLLM, and others. For instance, we can use Megatron-LM for training and vLLM to expedite inference.
4. 🎯**Flexible Parallel Strategies and Resource Allocation**: ChatLearn supports different parallel strategies for various model configurations, enabling the formulation of distinct parallel approaches tailored to each model's computational, memory, and communication characteristics. Additionally, ChatLearn features a flexible resource scheduling mechanism that accommodates exclusive or shared use of resources across models. Through its system scheduling policies, it facilitates efficient serial/parallel execution and optimized GPU memory sharing, enhancing overall performance and efficiency.
5. ⚡**High performance**: Compared to current state-of-the-art (SOTA) systems, ChatLearn achieves a 52% performance improvement at the 7B+7B(Policy+Reward) scale and a 137% improvement at the 70B+70B scale. Meanwhile, ChatLearn supports larger-scale alignment training, such as 300B+300B.

By providing a comprehensive and efficient framework, ChatLearn empowers researchers and practitioners to train RL for large language models with ease, scalability, and improved performance.

# Quick Start

Please refer to the [documentation](https://chatlearn.readthedocs.io/en/latest/) for a quick start.

1. [Environment and Code Setup](docs/en/installation.md)
2. [End-to-End GRPO Training Tutorial with FSDP + vLLM on Qwen3 Model](docs/en/tutorial/tutorial_grpo_fsdp.md)
3. [End-to-End Training Tutorial with LLaMA/LLaMA2 Model](docs/en/tutorial/tutorial_llama2.md)


# Performance

We compared the RLHF training throughput of models with different parameter scales, adopting an N+N model configuration where both the Policy model and the Reward model have the same number of parameters. We benchmarked against DeepSpeed-Chat and OpenRLHF with 7B and 70B model configurations. For the 8 GPU setup with a 7B+7B scale, we achieved a 115% speedup; for the 32 GPU setup with a 70B+70B scale, the speedup was 208%. The larger the scale, the more pronounced the acceleration effect becomes. Additionally, ChatLearn can support even larger-scale alignment training, such as at a 300B+300B scale.

![Compare Performance](docs/images/perf.png)

Note: The performance of DeepSpeed-Chat and OpenRLHF has already been optimized.

# Feature List

- Supports RLHF, DPO, OnlineDPO, GRPO, and user-defined RL training methods.
- Supports Megatron-LM, FSDP as the backend for training, and vLLM as the backend for inference.
- Supports independent configuration of parallel strategies for different models, and efficient parameter synchronization between models.
- Supports EMS (Efficient Memory Sharing) functionality, enabling efficient memory sharing between models.
- Supports resource types for models: GPU, CPU, such as defining a pure CPU-based Math Reward model.
- Support models with Megatron-Core format.

# Roadmap

The upcoming features for ChatLearn include:
- [x] Simplify Configuration Settings
- [x] Support tutorials for the RL training of MoE (Mixture of Experts) models
- [ ] Support for more models
- [ ] Performance Optimization
- [ ] Support for more RL algorithms

<br><br>
We welcome community partners to collaborate and contribute to the development, and welcome to join the DingTalk group: 98090003312 to participate in the discussion.
We are continuously hiring and welcome you to contact us or submit your resume to [email](mailto:wanglin.zj@alibaba-inc.com).
