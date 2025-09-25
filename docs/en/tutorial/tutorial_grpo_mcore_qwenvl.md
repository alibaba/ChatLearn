# End-to-End Qwen2.5-VL GRPO Training Tutorial with Mcore

This document provides instructions for end-to-end training using the ChatLearn, Mcore and vLLM framework, and the qwen2.5-vl 7B model.

## Environment Setup
1. Docker Image Preparation

We recommend running the following example in PAI [DSW](https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances/)/[DLC]( https://help.aliyun.com/zh/pai/user-guide/create-a-training-task). You need to use the following image to launch the instance.
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312
```

You can use a VPC address to accelerate image pulling. The image address should be adjusted based on the current region. For example, if you need to launch a DSW instance in Shanghai, you can use the following image `dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312`.

2. Code Preparation

```bash
git clone https://github.com/alibaba/ChatLearn.git
wget http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/csrc/megatron-patch-release/0922/Pai-Megatron-Patch.tar.gz
tar -xvf Pai-Megatron-Patch.tar
```

## Data & Model Preparation
## Data Preparation
We take [geo3k](https://hf-mirror.com/datasets/hiyouga/geometry3k) as exmaple.
```bash
# download dataset
mkdir -p dataset
export HF_ENDPOINT=https://hf-mirror.com

# data process
python chatlearn/data/data_preprocess/geo3k.py

# model preparation
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir pretrained_models/Qwen2.5-VL-7B-Instruct
```

## CKPT Conversion

Please check [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) for detailed ckpt conversion

Below codes show how to convert qwen2.5-vl 7B model ckpt.
```bash
CHATLEARN_ROOT=$(pwd)
cd ../Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen2_5_vl/run_8xH20.sh \
7B \
${CHATLEARN_ROOT}/pretrained_models/Qwen2.5-VL-7B-Instruct  \
${CHATLEARN_ROOT}/pretrained_models//Qwen2.5-VL-7B-Instruct-to-mcore \
false  \
true  \
bf16
```

## Training
You can run the following command to start training:

```bash
cd ${CHATLEARN_ROOT}

# vllm
bash scripts/mcore_vllm/train_mcore_vllm_qwen2_5_vl_7b_grpo.sh
```

## Using Wandb
If you want to use Wandb to log the training process, you need to modify the following configuration in [train_mcore_vllm_qwen2_5_vl_7b_grpo.sh](../../../scripts/mcore_vllm/train_mcore_vllm_qwen2_5_vl_7b_grpo.sh):

```bash
export WANDB_API_KEY="Your-Wandb-api-key"
```
Change the configuration to:
```bash
runtime_args.log_args_dict.enable_wandb=True
runtime_args.log_args_dict.wandb_project="Your-Wandb-Project-Name"
```