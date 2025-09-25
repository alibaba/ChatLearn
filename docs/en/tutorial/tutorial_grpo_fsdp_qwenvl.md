# Qwen2.5-VL End-to-End GRPO Training Tutorial with FSDP

This document provides instructions for end-to-end training using the ChatLearn, pytorch FSDP and vLLM framework, and the qwen2.5vl-7b model.

## Environment Setup
1. Docker Image Preparation

We recommend running the following example in PAI [DSW](https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances/)/[DLC]( https://help.aliyun.com/zh/pai/user-guide/create-a-training-task). You need to use the following image to launch the instance.
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312
```

You can use a VPC address to accelerate image pulling. The image address should be adjusted based on the current region. For example, if you need to launch a DSW instance in Shanghai, you can use the following image `dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312`.

2. Code Preparation

```bash
git clone https://github.com/alibaba/ChatLearn.git && cd ChatLearn
```

## Data Preparation
We take [geo3k](https://hf-mirror.com/datasets/hiyouga/geometry3k) as exmaple.
```bash
# download dataset
mkdir -p dataset
export HF_ENDPOINT=https://hf-mirror.com

# data process
python chatlearn/data/data_preprocess/geo3k.py
```

## Training
You can run the following command to start training:

### Qwen2.5VL-7B
Run this command on server with 8 GPUs
```bash
# download model weight
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir pretrained_models/Qwen2.5-VL-7B-Instruct

# vllm
bash scripts/fsdp_vllm/train_fsdp_vllm_qwen2_5_vl_7b_grpo.sh
```

## Using Wandb
If you want to use Wandb to log the training process, you need to modify the configuration with: 
```bash
export WANDB_API_KEY="Your-Wandb-api-key"
```
Change the configuration to:
```bash
runtime_args.log_args_dict.enable_wandb=True
runtime_args.log_args_dict.wandb_project="Your-Wandb-Project-Name"
```

## Model Conversion
Saving FSDP models is time-consuming. Chatlearn provides an offline model conversion feature, which converts FSDP-sharded checkpoints back to HuggingFace format. The script is as follows:
```bash
export CHATLEARN=$(pwd)
python chatlearn/offline_ckpt_converter.py \
    --hf_dir ${CHATLEARN}/Qwen2.5-VL-7B-Instruct/ \
    --ckpt_dir ${CHATLEARN}/output/qwen25vl-grpo-7b/save_model/policy_trainer \
    --save_dir ${CHATLEARN}/output/qwen25vl-grpo-7b/save_model/huggingface/ \
    --iter 200 \
    --groupgemm 0
```
If you are training an MoE model with groupgemm, please make sure to set:
```bash
   --groupgemm 1
```
This script will convert the final FSDP sharded model after training back into a HuggingFace model and save it in the path "${CHATLEARN}/output/qwen25vl-grpo-7b/save_model/huggingface/".