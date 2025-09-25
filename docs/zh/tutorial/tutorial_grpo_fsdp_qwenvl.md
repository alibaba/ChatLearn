# 基于 FSDP 的端到端 Qwen2.5VL GRPO训练流程

本文档提供使用 ChatLearn、PyTorch FSDP 和 vLLM 框架来对Qwen2.5-VL模型进行GRPO训练的快速开始指南。

## 环境配置
1. Docker镜像准备
我们建议在PAI [DSW](https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances/)/[DLC](https://help.aliyun.com/zh/pai/user-guide/create-a-training-task)中运行该示例，你需要填写如下镜像地址来启动实例：
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312
```

可以使用vpc地址来加速镜像拉取速度，需要根据当前region信息来更改镜像地址。比如，启动在上海的DSW实例，可以使用如下镜像`dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312`。

2. 代码准备

```bash
git clone https://github.com/alibaba/ChatLearn.git && cd ChatLearn
```

## 数据准备

以[geo3k](https://hf-mirror.com/datasets/hiyouga/geometry3k)数据集作为示例.
```bash
# 下载数据集
mkdir -p dataset

export HF_ENDPOINT=https://hf-mirror.com

# 数据集预处理
python chatlearn/data/data_preprocess/geo3k.py
```

## 训练
运行以下命令开始训练：

### Qwen2.5VL-7B
8卡机器运行如下命令
```bash
# 下载模型权重
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir pretrained_models/Qwen2.5-VL-7B-Instruct

# vllm
bash scripts/fsdp_vllm/train_fsdp_vllm_qwen2_5_vl_7b_grpo.sh
```

## 使用 Wandb 监控
如需使用 Wandb 记录训练过程，请修改对应脚本中的配置：

```bash
export WANDB_API_KEY="Your-Wandb-api-key"
```
将配置项改为：
```bash
runtime_args.log_args_dict.enable_wandb=True
runtime_args.log_args_dict.wandb_project="Your-Wandb-Project-Name"
```

## 模型转化
FSDP模型保存耗时较高，Chatlearn提供了离线模型转化功能，将FSDP保存的切片模型转化回huggingface模型。脚本如下：
```bash
export CHATLEARN=$(pwd)
python chatlearn/offline_ckpt_converter.py \
    --hf_dir ${CHATLEARN}/Qwen2.5-VL-7B-Instruct/ \
    --ckpt_dir ${CHATLEARN}/output/qwen25vl-grpo-7b/save_model/policy_trainer \
    --save_dir ${CHATLEARN}/output/qwen25vl-grpo-7b/save_model/huggingface/ \
    --iter 200 \
    --groupgemm 0
```
如果你使用groupgemm优化的moe模型训练，请确保设置：
```bash
   --groupgemm 1
```
这段脚本会将训练完成后的最后一个FSDP切片模型转化回HF模型，并保存在"${CHATLEARN}/output/qwen25vl-grpo-7b/save_model/huggingface/"路径下
