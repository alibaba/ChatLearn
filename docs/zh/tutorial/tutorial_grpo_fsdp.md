# 基于 FSDP 的端到端GRPO训练流程

本文档提供使用 ChatLearn、PyTorch FSDP 和 vLLM 框架来对Qwen3模型进行GRPO训练的快速开始指南。

## 环境配置
1. Docker镜像准备
建议使用以下预构建镜像:
```bash
dsw-registry.cn-shanghai.cr.aliyuncs.com/pai/modelscope:1.23.0-pytorch2.5.1-gpu-py310-cu124-ubuntu22.04
```
如果在 PAI DLC/DSW 环境中训练，可直接使用标签为 `modelscope:1.23.0-pytorch2.5.1-gpu-py310-cu124-ubuntu22.04`的镜像.

2. 代码准备与依赖安装

```bash
pip install vllm==0.6.6 cupy-cuda12x==13.4.1 wandb==0.19.11 ray==2.40.0 transformers==4.51.3
git clone https://github.com/alibaba/ChatLearn.git && cd ChatLearn
```

## 数据准备
以[MATH-lighteval](https://www.modelscope.cn/datasets/AI-ModelScope/MATH-lighteval)数据集作为示例.
```bash
# 下载数据集
mkdir -p dataset
modelscope download --dataset AI-ModelScope/MATH-lighteval --local_dir dataset/MATH-lighteval
# 数据集预处理
python examples/fsdp/data/data_preprocess/math_lighteval.py --input_dir dataset/MATH-lighteval --local_dir dataset/MATH-lighteval
# 下载模型权重
modelscope download --model Qwen/Qwen3-8B --local_dir Qwen3-8B
```

## 训练
运行以下命令开始训练：

```bash
bash examples/fsdp/scripts/train_grpo_qwen3.sh
```

## 使用 Wandb 监控
如需使用 Wandb 记录训练过程，请修改[train_grpo_qwen3.sh](../../../examples/fsdp/scripts/train_grpo_qwen3.sh)中的配置：

```bash
export enable_wandb=True
export wandb_project="Your-Wandb-Project-Name"
export WANDB_API_KEY="Your-Wandb-api-key"
```