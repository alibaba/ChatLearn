# Environment and Code Preparation

## 1. Image Preparation

ChatLearn supports vLLM and SGLang as backend frameworks for Rollout generation. Depending on the chosen Rollout backend, you can select the appropriate Docker image for your experiments.

### vLLM

You can prepare the image by referring to [Dockerfile.torch2.6.0.vllm085](https://github.com/alibaba/ChatLearn/blob/main/docker/torch/Dockerfile.torch2.6.0.vllm085). Alternatively, you can directly pull and use the following image:

```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312
```

### SGLang

We will provide SGLang-related Docker images in the future.

## 2. Code Preparation

```bash
# Download ChatLearn code
git clone https://github.com/alibaba/ChatLearn.git
```

If you choose Megatron as the training framework, you need to download [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch).

```bash
# Download Megatron-LM
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

> If you encounter network connectivity issues with GitHub, you can alternatively download our pre-prepared `Pai-Megatron-Patch` archive using the following command:  
`wget https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/csrc/Pai-Megatron-Patch.tar && tar -xvf Pai-Megatron-Patch.tar`

## 3. Running Reinforcement Learning Experiments

ChatLearn supports FSDP and Megatron as training backends. Please refer to the following tutorials for detailed instructions:

- [End-to-End GRPO Training with FSDP](https://github.com/alibaba/ChatLearn/blob/main/docs/en/tutorial/tutorial_grpo_fsdp.md)
- [End-to-End GRPO Training with Mcore](https://github.com/alibaba/ChatLearn/blob/main/docs/en/tutorial/tutorial_grpo_mcore.md)