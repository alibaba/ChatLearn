# 环境和代码准备

1. 镜像准备

推荐参考 `https://github.com/alibaba/ChatLearn/tree/master/docker/ngc/Dockerfile.ngc23.09` 准备镜像。
如果在 PAI DLC 环境上训练，推荐使用我们准备好的镜像：

```bash
registry.cn-wulanchabu.aliyuncs.com/pai-dlc/pytorch-training:2.1.0-gpu-py3.10-cu12.2-ngc23.09-ubuntu22.04
```

2. 代码准备: 用户需要下载 `ChatLearn` 框架代码。

```
# 下载 ChatLearn 代码
git clone https://github.com/alibaba/ChatLearn.git
```

3. 如果您需要运行基于 Megatron-LM 框架的 RLHF 训练程序，您也需要下载 `Megatron-LM` 代码。

```
# 下载 Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
```
