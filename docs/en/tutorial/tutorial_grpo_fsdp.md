# End-to-End GRPO Training Tutorial with FSDP

This document provides instructions for end-to-end training using the ChatLearn, pytorch FSDP and vLLM framework, and the qwen3 model.

## Environment Setup
1. Docker Image Preparation

We recommend running the following example in PAI [DSW](https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances/)/[DLC](https://help.aliyun.com/zh/pai/user-guide/create-a-training-task?spm=a2c4g.11186623.help-menu-30347.d_3_3_5_5.2dfb1925l3QjwG). You need to use the following image to launch the instance.
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312
```

You can use a VPC address to accelerate image pulling. The image address should be adjusted based on the current region. For example, if you need to launch a DSW instance in Shanghai, you can use the following image `dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312`.

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
python chatlearn/data/data_preprocess/math_lighteval.py --input_dir dataset/MATH-lighteval --local_dir dataset/MATH-lighteval
# download model weight
modelscope download --model Qwen/Qwen3-8B --local_dir Qwen3-8B
```

## Training
You can run the following command to start training:

```bash
bash scripts/train_fsdp_vllm_qwen3_8b_grpo.sh
```

## Using Wandb
If you want to use Wandb to log the training process, you need to modify the following configuration in [train_fsdp_vllm_qwen3_8b_grpo.sh](../../../scripts/train_fsdp_vllm_qwen3_8b_grpo.sh):

```bash
export WANDB_API_KEY="Your-Wandb-api-key"
```
Change the configuration to:
```bash
runtime_args.log_args_dict.enable_wandb=True
runtime_args.log_args_dict.wandb_project="Your-Wandb-Project-Name"
```