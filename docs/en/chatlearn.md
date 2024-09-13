# ChatLearn: A flexible and efficient training framework for large-scale alignment

ChatLearn aims to provide a flexible and user-friendly platform for alignment training based on Large Language Models (LLMs) such as ChatGPT.

## Introduction

ChatGPT, developed by OpenAI, is a chatbot model based on a large language model (LLM) that has gained popularity and widespread adoption for its impressive conversational capabilities. The success of ChatGPT can be attributed to the new training paradigm called Reinforcement Learning from Human Feedback (RLHF). RLHF optimizes language models based on human feedback using reinforcement learning techniques.

![RLHFFlow](../images/rlhf.png)

Unlike traditional deep learning training, which involves iterations and optimization of a single model, RLHF and similar training paradigms necessitate the computation and data interaction of multiple large models. This poses numerous challenges in building a user-friendly and efficient training system.

1. **Programming Interface**: How to design a universal and flexible programming interface that allows users to focus on the modeling of individual models while also providing flexible control over the interaction between models.
2. **Distributed Acceleration backends**: As the scale of models increases, users often resort to distributed computing and acceleration backends such as Megatron-LM, DeepSpeed and vLLM to improve performance. Integrating these acceleration backends into a multi-model computation framework requires careful consideration and design.
3. **Parallel Strategies**: Different models may possess distinct computational characteristics. For instance, models solely used for inference and those intended for training exhibit variations in terms of memory usage and computational requirements. Additionally, the most suitable parallel strategy may differ for each model. Consequently, a framework should enable the configuration of different parallel strategies for different models to maximize overall performance.
4. **Resource Allocation**: How to flexibly allocate resources to multiple models to achieve efficient concurrent scheduling and execution.

To address these challenges, we propose a novel alignment training framework called ChatLearn. ChatLearn abstracts the computation logic of models, decoupling the models from the computation backend and parallel strategies. It provides a flexible resource scheduling mechanism that supports flexible resource allocation and parallel scheduling strategies. Chatlearn has the following advantages:

1. **User-friendly programming interface**: Users can focus on programming individual models by wrapping a few functions, while the system takes care of resource scheduling, data and control flow transmission, and distributed execution.
2. **Highly Scalable Training Methodology**: ChatLearn offers alignment training such as RLHF, DPO, OnlineDPO and GRPO, while also supporting user-defined execution flows for models, enabling a highly convenient and customizable training process.
3. **Diverse Distributed Acceleration Engines**: Users can model their models using different computing backends such as Megatron-LM, DeepSpeed, vLLM, etc. Users can also combine different backends, for example, using Megatron-LM to accelerate training and vLLM to speed up inference.
4. **Flexible Parallel Strategies and Resource Allocation**: ChatLearn supports different parallel strategies for various model configurations, enabling the formulation of distinct parallel approaches tailored to each model's computational, memory, and communication characteristics. Additionally, ChatLearn features a flexible resource scheduling mechanism that accommodates exclusive or shared use of resources across models. Through its system scheduling policies, it facilitates efficient serial/parallel execution and optimized GPU memory sharing, enhancing overall performance and efficiency.
5. **High performance**: Compared to current state-of-the-art (SOTA) systems, ChatLearn achieves a 52% performance improvement at the 7B+7B(Policy+Reward) scale and a 137% improvement at the 70B+70B scale. Meanwhile, ChatLearn supports larger-scale alignment training, such as 300B+300B.

By providing a comprehensive and efficient framework, ChatLearn empowers researchers and practitioners to train large-scale alignment models with ease, scalability, and improved performance.

## Technical Architecture

![arch](../images/arch.png)

**API:** ChatLearn offers training for alignment through methods such as RLHF, DPO, Online DPO, and GRPO. It also supports the customization of the model execution flow by users to implement their own training processes. Additionally, ChatLearn provides a module abstraction, enabling users to encapsulate different computational backends by inheriting from MegatronModule, DeepSpeedModule, or VLLMModule. Through the use of YAML files, ChatLearn facilitates flexible configuration of models and parallel strategies by specifying different hyperparameters and parallelization tactics for alignment training and various model configurations.

**Scheduler:** ChatLearn introduces the abstraction of DistActor to support distributed model training or inference. The DistActor inherits the state management of the Ray actor and the isolation between workers, while at the same time breaking through the limitation that Ray actors cannot span machines. Through DistActor, ChatLearn is capable of supporting model inference and training at any scale. Simultaneously, the ChatLearn Scheduler achieves hardware-aware affinity scheduling by partitioning cluster Resource Groups and adjusting scheduling policies. ChatLearn also supports flexible resource allocation, accommodating strategies such as resource reuse, exclusivity, or partial sharing among models. This allows for the maximization of training efficiency given a certain number of resources.

**Executor:** The ChatLearn Executor divides the alignment training process into three primary modules: `Environment`, `Trainer`, and `Evaluator`. The `Environment` is responsible for the concurrent execution and management of inference model and data, the `Trainer` handles the corresponding training module, and the `Evaluator` oversees the assessment of model performance. Additionally, the Executor manages data transmission and parameter synchronization.

**Backend:** Thanks to the well-designed programming interface abstractions of ChatLearn, users can easily integrate various distributed acceleration backends through simple encapsulation, such as Megatron-LM, DeepSpeed and vLLM.

**Optimization:** ChatLearn also supports various optimizations such as computation, memory, and communication. It accelerates training through a combination of various parallel strategies, and speeds up inference by leveraging techniques like paged attention and continuous batching. The system efficiently reuses GPU memory and reduces overall resource requirements through the implementation of EMS (Efficient Memory Sharing) technology. Additionally, it employs grouping broadcast technology to facilitate efficient parameter synchronization between training and inference models.


## Quick Start

Please refer to the [Documentation](https://chatlearn.readthedocs.io/en/latest/) for a quick start guide.

1. [Environment and Code Setup](installation.md) 
2. [End-to-End Training Tutorial with Llama/Llama2 Model](tutorial/tutorial_llama2.md)

## Performance

We compared the RLHF training throughput of models with different parameter scales, adopting an N+N model configuration where both the Policy model and the Reward model have the same number of parameters. We benchmarked against DeepSpeed-Chat and OpenRLHF with 7B and 70B model configurations. For the 8 GPU setup with a 7B+7B scale, we achieved a 115% speedup; for the 32 GPU setup with a 70B+70B scale, the speedup was 208%. The larger the scale, the more pronounced the acceleration effect becomes. Additionally, ChatLearn can support even larger-scale alignment training, such as at a 300B+300B scale.

![compare perf](../images/perf.png)

Note: The performance of DeepSpeed-Chat and OpenRLHF has already been optimized.

## Roadmap

The upcoming features for ChatLearn include:
- [ ] Support models with Megatron-Core format
- [ ] Support the alignment training for MoE (Mixture of Experts) models
- [ ] Integration with DeepSpeed as a training backend
- [ ] Support for more models
- [ ] Performance Optimization
- [ ] Support for more alignment algorithms

<br><br>

We welcome community members to collaborate and contribute to the development of ChatLearn.


## Reference

1. Megatron-LM: https://github.com/NVIDIA/Megatron-LM
2. DeepSpeed-Chat: https://github.com/microsoft/DeepSpeedExamples/tree/master/applications/DeepSpeed-Chat
3. OpenRLHF: https://github.com/OpenRLHF/OpenRLHF
