# 基于 FSDP 的端到端GRPO训练流程

本文档提供使用 ChatLearn、PyTorch FSDP 和 vLLM 框架来对Qwen3模型进行GRPO训练的快速开始指南。

## 环境配置
1. Docker镜像准备
我们建议在PAI [DSW](https://help.aliyun.com/zh/pai/user-guide/create-and-manage-dsw-instances/)/[DLC](https://help.aliyun.com/zh/pai/user-guide/create-a-training-task?spm=a2c4g.11186623.help-menu-30347.d_3_3_5_5.2dfb1925l3QjwG)中运行该示例，你需要填写如下镜像地址来启动实例：
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312
```

可以使用vpc地址来加速镜像拉取速度，需要根据当前region信息来更改镜像地址。比如，启动在上海的DSW实例，可以使用如下镜像`dsw-registry-vpc.cn-shanghai.cr.aliyuncs.com/pai-training-algorithm/chatlearn:torch2.6.0-vllm0.8.5-ubuntu24.04-cuda12.6-py312`。

2. 代码准备

```bash
git clone https://github.com/alibaba/ChatLearn.git && cd ChatLearn
```

## 数据准备
以[MATH-lighteval](https://www.modelscope.cn/datasets/AI-ModelScope/MATH-lighteval)数据集作为示例.
```bash
# 下载数据集
mkdir -p dataset
modelscope download --dataset AI-ModelScope/MATH-lighteval --local_dir dataset/MATH-lighteval
# 数据集预处理
python chatlearn/data/data_preprocess/math_lighteval.py --input_dir dataset/MATH-lighteval --local_dir dataset/MATH-lighteval
# 下载模型权重
modelscope download --model Qwen/Qwen3-8B --local_dir Qwen3-8B
```

## 训练
运行以下命令开始训练：

```bash
bash scripts/train_fsdp_vllm_qwen3_8b_grpo.sh
```

## 使用 Wandb 监控
如需使用 Wandb 记录训练过程，请修改[train_grpo_qwen3.sh](../../../scripts/train_fsdp_vllm_qwen3_8b_grpo.sh)中的配置：

```bash
export WANDB_API_KEY="Your-Wandb-api-key"
```
将配置项改为：
```bash
runtime_args.log_args_dict.enable_wandb=True
runtime_args.log_args_dict.wandb_project="Your-Wandb-Project-Name"
```