# Environment and Code Setup

1. Docker Image Preparation

It is recommended to refer to `https://github.com/alibaba/ChatLearn/tree/master/docker/ngc/Dockerfile.ngc22.10` for docker image preparation.

2. Code Preparation: Users need to download the ChatLearn framework code.

```
# Clone ChatLearn code
git clone https://github.com/alibaba/ChatLearn.git
```

3. If you need to run the RLHF training program based on the Megatron-LM framework, you also need to download the modified `Megatron-LM-ChatLearn` code that supports ChatLearn training.

```
# Clone Megatron-LM-ChatLearn
git clone https://github.com/alibaba/Megatron-LM-ChatLearn.git
```