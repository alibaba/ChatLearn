# 环境和代码准备

## 1.镜像准备

ChatLearn支持vLLM和SGLang作为Rollout生成的后端框架，可以根据不同的Rollout后端框架，选择不同镜像进行实验。

### vLLM
可以参考 [Dockerfile.torch2.6.0.vllm085](https://github.com/alibaba/ChatLearn/blob/main/docker/torch/Dockerfile.torch2.6.0.vllm085) 准备镜像。也可以直接拉取如下镜像地址直接进行使用。

```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312
```

### SGLang

我们会在未来提供SGLang相关镜像。

## 2. 代码准备

```
# 下载 ChatLearn 代码
git clone https://github.com/alibaba/ChatLearn.git 
```

如果您选择Megatron作为训练框架，您需要下载[Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch)。

```
# 下载 Megatron-LM
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

> 如果github存在网络阻塞问题，您可以选择通过如下命令直接下载我们预先准备好的`Pai-Megatron-Patch`压缩包进行使用。`wget https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/csrc/Pai-Megatron-Patch.tar && tar -xvf Pai-Megatron-Patch.tar`

## 3. 进行强化学习实验

ChatLearn支持FSDP和Megatron作为训练后端，可以分别参考如下教程进行实验：

- [基于 FSDP 的端到端GRPO训练流程](https://github.com/alibaba/ChatLearn/blob/main/docs/zh/tutorial/tutorial_grpo_fsdp.md)
- [基于 Mcore 的端到端GRPO训练流程](https://github.com/alibaba/ChatLearn/blob/main/docs/zh/tutorial/tutorial_grpo_mcore.md)
