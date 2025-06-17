# End-to-End GRPO Training Tutorial with Mcore

This document provides instructions for end-to-end training using the ChatLearn, Mcore and vLLM framework, and the qwen2.5 model.

## Environment Setup
1. Docker Image Preparation

We recommend running the following example in PAI [DSW](https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances/)/[DLC](https://help.aliyun.com/zh/pai/user-guide/create-a-training-task?spm=a2c4g.11186623.help-menu-30347.d_3_3_5_5.2dfb1925l3QjwG). You need to use the following image to launch the instance.
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312
```

You can use a VPC address to accelerate image pulling. The image address should be adjusted based on the current region. For example, if you need to launch a DSW instance in Shanghai, you can use the following image `dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312`.

2. Code Preparation

```bash
git clone https://github.com/NVIDIA/Megatron-LM.git
cd Megatron-LM && git checkout 6ba97dd37150a6bfba03d31808674211cf2a4d0d
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
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir Qwen2.5-3B-Instruct
```

## CKPT Conversion

Please check [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) for detailed ckpt conversion
dist_ckpt conversion please refer to https://github.com/alibaba/Pai-Megatron-Patch/tree/main/toolkits/distributed_checkpoints_convertor

run `hf2mcore_qwen2.5_convertor.sh` script.
```
MODEL_SIZE=$1                  # model parameters：0.5B/1.5B/3B/7B/14B/32B/72B
SOURCE_CKPT_PATH=$2            # source path
TARGET_CKPT_PATH=$3            # target path
TP=$4                          # tensor model parallel size
PP=$5                          # pipeline model parallel size
PR=$6                          # precision
USE_TE=$7                      # whether use transformer engine
mg2hf=$8                       # whether run mg2hf
HG_CKPT_PATH=$9                # huggingface ckpt path
```

Below codes show how to convert qwen2.5 7B model ckpt.
```bash
wget https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/csrc/Pai-Megatron-Patch.tar
tar -xvf Pai-Megatron-Patch.tar
cd ~/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/qwen
bash hf2mcore_qwen2.5_convertor.sh \
3B \
/mnt/qwen-ckpts/Qwen2.5-3B-Instruct  \
/mnt/qwen-ckpts/Qwen2.5-3B-Instruct-hf-to-mcore-tp2-pp1 \
2  \
1  \
bf16 \
true \
false 
```

## Training
You can run the following command to start training:

```bash
export MEGATRON_PATH="your megatron path"
bash scripts/train_megatron_vllm_qwen2.5_3b_grpo.sh
```

## Using Wandb
If you want to use Wandb to log the training process, you need to modify the following configuration in [train_grpo_qwen3.sh](../../../scripts/train_fsdp_vllm_qwen3_8b_grpo.sh):

```bash
export WANDB_API_KEY="Your-Wandb-api-key"
```
Change the configuration to:
```bash
runtime_args.log_args_dict.enable_wandb=True
runtime_args.log_args_dict.wandb_project="Your-Wandb-Project-Name"
```