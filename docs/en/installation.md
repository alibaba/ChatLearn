# Environment and Code Setup

1. Docker Image Preparation

It is recommended to refer to `https://github.com/alibaba/ChatLearn/tree/master/docker/ngc/Dockerfile.ngc23.09` for preparing the docker image.
If you're training on the PAI DLC environment, we suggest using the pre-built image provided below:

```bash
registry.cn-wulanchabu.aliyuncs.com/pai-dlc/pytorch-training:2.1.0-gpu-py3.10-cu12.2-ngc23.09-ubuntu22.04
```

2. Code Preparation: Users need to download the ChatLearn framework code.

```
# Clone ChatLearn code
git clone https://github.com/alibaba/ChatLearn.git
```

3. If you need to run the RLHF training program based on the Megatron-LM framework, you also need to download the `Megatron-LM` code.

```
# Clone Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
```
