# 基于 Mcore 的端到端GRPO训练流程

本文档提供使用 ChatLearn、Mcore 和 SGLANG 框架来对Qwen3-next进行GRPO训练的快速开始指南。

## 开发环境配置
建议在PAI平台DSW环境中基于nvcr.io/nvidia/pytorch:24.12-py3来构建镜像。
```bash

#安装SGLANG，注意这将移除NGC自带的Pytorch，而自动重新安装pytorch==2.8.0
pip install --no-cache-dir "sglang[all]==0.5.2"  -i https://mirrors.aliyun.com/pypi/simple/ 

#添加SGLANG PATCH
wget https://gist.github.com/lostkevin/9b668c24de6f0e9974c9ad069ef03ed9
cp memory_pool.py /usr/local/lib/python3.12/dist-packages/sglang/srt/mem_cache/


#安装Chatlearn的依赖包
pip install transformers==4.57.1 modelscope==1.30.0 tensordict==0.10.0 torchdata==0.11.0 codetiming==1.4.0 blobfile==3.0.0 numpy==1.26.4 accelerate==1.10.0 wandb==0.19.11 datasets==3.6.0 grpcio==1.71.0 omegaconf==2.3.0  hydra-core==1.3.2 msgspec==0.19.0 mathruler==0.1.0 pylatexenc==2.10 langgraph==0.6.6 ray[default]==2.46.0 -i https://mirrors.aliyun.com/pypi/simple/ 

#由于安装VLLM会重新安装pytorch，因此需要重新安装flash attention以及apex
pip uninstall -y flash_attn && pip install https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/csrc/flash-attention/torch2.6.0-cu12x/flash_attn-2.4.2-cp312-cp312-linux_x86_64.whl --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ 

pip uninstall -y apex && pip install https://pai-vision-data-hz.oss-cn-zhangjiakou.aliyuncs.com/csrc/apex/torch2.6.0-cuda12x/apex-0.1-cp312-cp312-linux_x86_64.whl --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ 


#升级Transformer Engine
pip uninstall -y transformer-engine transformer-engine-cu12 transformer-engine-torch
git clone --recursive https://github.com/NVIDIA/TransformerEngine.git
cd TransformerEngine
git submodule update --init --recursive
git checkout release_v2.7
export CUDNN_PATH=/usr/local/lib/python3.12/dist-packages/nvidia/cudnn/
cp /usr/local/lib/python3.12/dist-packages/nvidia/cudnn/include/*  /usr/local/cuda/include/
python setup.py bdist_wheel  -vvv
cd dist
export NVTE_FRAMEWORK=pytorch 
pip install transformer_engine-2.7.0-cp312-cp312-linux_x86_64.whl --no-cache-dir -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com

#安装mamba-ssm依赖
pip install --no-build-isolation  "mamba-ssm" -i https://mirrors.aliyun.com/pypi/simple/

#安装causal-conv1d依赖
git clone https://github.com/Dao-AILab/causal-conv1d.git
cd causal-conv1d
git checkout v1.5.2
export CAUSAL_CONV1D_FORCE_BUILD=TRUE
python setup.py bdist_wheel  -vvv
cd dist
export NVTE_FRAMEWORK=pytorch 
pip install causal_conv1d-1.5.2-cp312-cp312-linux_x86_64.whl --no-cache-dir --no-build-isolation -i https://mirrors.aliyun.com/pypi/simple/ --trusted-host mirrors.cloud.aliyuncs.com

# 安装flash-linear-attention
pip install --no-build-isolation  flash-linear-attention -i https://mirrors.aliyun.com/pypi/simple/

```
## 代码准备

```bash
git clone https://github.com/alibaba/ChatLearn.git
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

## 数据&模型准备
以[MATH-lighteval](https://www.modelscope.cn/datasets/AI-ModelScope/MATH-lighteval)数据集作为示例.
```bash
cd ChatLearn
# 下载数据集
mkdir -p dataset
modelscope download --dataset AI-ModelScope/MATH-lighteval --local_dir dataset/MATH-lighteval
# preprocess dataset
python chatlearn/data/data_preprocess/math_lighteval.py --input_dir dataset/MATH-lighteval --local_dir dataset/MATH-lighteval
# download model weight
modelscope download --model Qwen/Qwen3-Next-80B-A3B-Instruct --local_dir Qwen3-Next-80B-A3B-Instruct

```

## 模型转换
使用下述脚本将Moonlight和DeepSeek-V3的Huggingface格式的模型转换到MCore格式
```bash
CHATLEARN_ROOT=$(pwd)
cd ../Pai-Megatron-Patch/toolkits/distributed_checkpoints_convertor
bash scripts/qwen3_next/run_8xH20.sh  \
A3B \
${CHATLEARN_ROOT}/pretrained_models/Qwen3-Next-80B-A3B-Instruct  \
${CHATLEARN_ROOT}/pretrained_models/Qwen3-Next-80B-A3B-Instruct-to-mcore \
false  \
true  \
bf16

```

## Qwen3-Next强化学习训练以及训练稳定性指引
运行以下命令可以对Qwen3-Next进行GRPO训练：

```bash
cd ${CHATLEARN_ROOT}
bash scripts/mcore_sglang/train_mcore_sglang_qwen3_next_grpo.sh
```

## 使用 Wandb 监控
如需使用 Wandb 记录训练过程，请参考其他最佳实践进行修改。
