# 基于 Mcore 的Qwen2.5-VL 端到端GRPO训练流程

本文档提供使用 ChatLearn、Mcore 和 vLLM 框架来对Qwen2.5-VL 7B模型进行GRPO训练的快速开始指南。

## 环境配置
1. Docker镜像准备
我们建议在PAI [DSW](https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances/)/[DLC](https://help.aliyun.com/zh/pai/user-guide/create-a-training-task)中运行该示例，你需要填写如下镜像地址来启动实例：
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312
```

可以使用vpc地址来加速镜像拉取速度，需要根据当前region信息来更改镜像地址。比如，启动在上海的DSW实例，可以使用如下镜像`dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312`。

2. 代码准备

```bash
git clone https://github.com/alibaba/ChatLearn.git
wget http://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/csrc/megatron-patch-release/0922/Pai-Megatron-Patch.tar.gz
tar -xvf Pai-Megatron-Patch.tar
```

## 数据&模型准备

以[geo3k](https://hf-mirror.com/datasets/hiyouga/geometry3k)数据集作为示例.
```bash
# 下载数据集
mkdir -p dataset

export HF_ENDPOINT=https://hf-mirror.com

# 数据集预处理
python chatlearn/data/data_preprocess/geo3k.py

# 下载模型权重
modelscope download --model Qwen/Qwen2.5-VL-7B-Instruct --local_dir pretrained_models/Qwen2.5-VL-7B-Instruct
```

## 模型转换
使用下述脚本将7B量级的Qwen2.5-VL的Huggingface格式的模型转换到MCore格式

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

## 训练
运行以下命令开始训练：

```bash
cd ${CHATLEARN_ROOT}

# vllm
bash scripts/mcore_vllm/train_mcore_vllm_qwen2_5_vl_7b_grpo.sh
```

## 使用 Wandb 监控
如需使用 Wandb 记录训练过程，请修改[train_mcore_vllm_qwen2_5_vl_7b_grpo.sh](../../../scripts/mcore_vllm/train_mcore_vllm_qwen2_5_vl_7b_grpo.sh)中的配置：

```bash
export WANDB_API_KEY="Your-Wandb-api-key"
```
将配置项改为：
```bash
runtime_args.log_args_dict.enable_wandb=True
runtime_args.log_args_dict.wandb_project="Your-Wandb-Project-Name"
```