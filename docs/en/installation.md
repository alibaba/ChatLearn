# Environment and Code Setup

1. Docker Image Preparation

It is recommended to refer to `https://github.com/alibaba/ChatLearn/tree/master/docker/torch/Dockerfile.torch2.3.0` for preparing the docker image.
If you're training on the PAI DLC/DSW environment, we suggest using the pre-built image provided below:

```bash
registry.cn-wulanchabu.aliyuncs.com/pai-dlc/pytorch-training:2.4.0-gpu-py3.10-cu12.5-ngc24.06-ubuntu22.04
```

2. Code Preparation: Users need to download the ChatLearn framework code.

```
# Clone ChatLearn code
git clone https://github.com/alibaba/ChatLearn.git
```

3. If you need to run the alignment training program based on the Megatron-LM framework, you also need to download the `Megatron-LM` code.

```
# Clone Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
git checkout 5161b1689
```
