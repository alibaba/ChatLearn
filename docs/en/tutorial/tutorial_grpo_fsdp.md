# End-to-End GRPO Training Tutorial with FSDP

This document provides instructions for end-to-end training using the ChatLearn, pytorch FSDP and vLLM framework, and the qwen3 model.

## Environment Setup
1. Docker Image Preparation

We recommend running the following example in PAI DSW/DLC. You need to use the following image to launch the instance.
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.5.1-vllm0.6.6-ubuntu22.04-cuda12.6-py310
```

You can use a VPC address to accelerate image pulling. The image address should be adjusted based on the current region. For example, if you need to launch a DSW instance in Shanghai, you can use the following image `dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.5.1-vllm0.6.6-ubuntu22.04-cuda12.6-py310`.

2. Code Preparation

```bash
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
modelscope download --model Qwen/Qwen3-8B --local_dir Qwen3-8B
```

## Training
You can run the following command to start training:

```bash
bash examples/fsdp/scripts/train_grpo_qwen3.sh
```

## Using Wandb
If you want to use Wandb to log the training process, you need to modify the following configuration in [train_grpo_qwen3.sh](../../../examples/fsdp/scripts/train_grpo_qwen3.sh):

```bash
export enable_wandb=True
export wandb_project="Your-Wandb-Project-Name"
export WANDB_API_KEY="Your-Wandb-api-key"
```