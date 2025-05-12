# End-to-End GRPO Training Tutorial with FSDP

This document provides instructions for end-to-end training using the ChatLearn, pytorch FSDP and vLLM framework, and the qwen2.5 model.

## Environment Setup
1. Docker Image Preparation
We suggest using the pre-build image provided below:
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai/modelscope:1.23.0-pytorch2.5.1-gpu-py310-cu124-ubuntu22.04
```
If you're training on the PAI DLC/DSW environment, you can use the image with tag `modelscope:1.23.0-pytorch2.5.1-gpu-py310-cu124-ubuntu22.04`.

2. Code Preparation & Installing Additional Dependencies

```bash
pip install vllm==0.6.6 cupy-cuda12x==13.4.1 wandb==0.19.11 ray==2.40.0
git clone https://github.com/alibaba/ChatLearn.git && cd ChatLearn
```

## Data Preparation
We take [MATH-lighteval](https://www.modelscope.cn/datasets/AI-ModelScope/MATH-lighteval) as exmaple.
```bash
# download dataset
mkdir -p dataset
modelscope download --dataset AI-ModelScope/MATH-lighteval --local_dir dataset/MATH-lighteval
# preprocess dataset
python examples/fsdp/data/data_preprocess/math_lighteval.py --input_dir dataset/MATH-lighteval --local_dir dataset/MATH-lighteval
# download model weight
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir Qwen2.5-7B-Instruct
```

## Training
You can run the following command to start training:

```bash
bash examples/fsdp/scripts/train_grpo_qwen.sh
```

## Using Wandb
If you want to use Wandb to log the training process, you need to modify the following configuration in [train_grpo_qwen.sh](examples/fsdp/scripts/train_grpo_qwen.sh):

```bash
export enable_wandb=True
export wandb_project="Your-Wandb-Project-Name"
export WANDB_API_KEY="Your-Wandb-api-key"
```