# Qwen3-VL End-to-End GRPO Training Tutorial with FSDP

This document provides instructions for end-to-end training using the ChatLearn, pytorch FSDP and vLLM framework, and the qwen3vl-8b model.

## Environment Setup
1. Docker Image Preparation

We recommend running the following example in PAI [DSW](https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances/)/[DLC]( https://help.aliyun.com/zh/pai/user-guide/create-a-training-task). You need to use the following image to launch the instance.
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.8.0-sglang0.5.3-ubuntu24.04-cuda12.6-py312
```

You can use a VPC address to accelerate image pulling. The image address should be adjusted based on the current region. For example, if you need to launch a DSW instance in Shanghai, you can use the following image `dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.8.0-sglang0.5.3-ubuntu24.04-cuda12.6-py312`.

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

### Qwen3VL-8B
Run this command on server with 8 GPUs
MOE model is also supported
```bash
# download model weight
modelscope download --model Qwen/Qwen3-VL-8B-Instruct --local_dir pretrained_models/Qwen3-VL-8B-Instruct

bash scripts/fsdp_sglang/train_fsdp_sglang_qwen3_vl_8b_grpo.sh
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
    --hf_dir ${CHATLEARN}/Qwen3-VL-8B-Instruct/ \
    --ckpt_dir ${CHATLEARN}/output/qwen3vl-grpo-8b/save_model/policy_trainer \
    --save_dir ${CHATLEARN}/output/qwen3vl-grpo-8b/save_model/huggingface/ \
    --iter 200 \
    --groupgemm 0
```
If you are training an MoE model with groupgemm, please make sure to set:
```bash
   --groupgemm 1
```
This script will convert the final FSDP sharded model after training back into a HuggingFace model and save it in the path "${CHATLEARN}/output/qwen3vl-grpo-8b/save_model/huggingface/".