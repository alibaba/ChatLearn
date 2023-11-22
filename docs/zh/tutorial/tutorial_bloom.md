# 基于 Bloom 模型的端到端训练教程

本文档介绍基于 ChatLearn, Megatron-LM 框架和 Bloom 模型的训练流程。包含三阶段的训练：SFT, Reward 和 RLHF 训练。


**以下是这个 Tutorial 脚本中使用的通用环境变量集合：**

| ENV | 含义 |
| --- | --- |
| `CHATLEARN` | ChatLearn 代码仓库 clone 存放的位置 [https://github.com/alibaba/ChatLearn.git](https://github.com/alibaba/ChatLearn.git) |
| `MEGATRON` | Megatron-LM-ChatLearn 代码仓库 clone 存放的位置 [https://github.com/alibaba/Megatron-LM-ChatLearn.git](https://github.com/alibaba/Megatron-LM-ChatLearn.git) |
| `DATASET_ROOT` | 存放SFT/Reward/RLHF训练数据集合的根目录 |
| `TOKENIZER_PATH` | Tokenizer 使用的 vocab_file 所在的文件夹 |


## Setup: 镜像、代码、数据准备

### 镜像

推荐参考 `https://github.com/alibaba/ChatLearn/tree/master/docker/ngc/Dockerfile.ngc22.10` 准备镜像。
如果在 PAI DLC 环境上训练，推荐使用我们准备好的镜像：

```bash
registry.cn-wulanchabu.aliyuncs.com/pai-dlc/pytorch-training:1.13.0-gpu-py3.8-cu11.8-ngc22.10-ubuntu20.04
```

### 代码

在这个示例中，我们需要下载以下相关代码。

```bash
# 下载为支持 Bloom 模型的 Megatron-LM-ChatLearn
git clone -b v0.1.0 https://github.com/alibaba/Megatron-LM-ChatLearn.git
# 下载ChatLearn代码
git clone -b v0.1.0 https://github.com/alibaba/ChatLearn.git
```

### 数据

请参考 [三阶段数据](data.md) 准备好您的训练数据。


## Step1: SFT

SFT 指的是使用有标注的对话数据来微调预训练语言模型的过程。在这个示例中，我们需要下载预训练的模型，然后开始一个简单的 SFT 训练示例。

### 下载和转化预训练模型

若使用来自于 HuggingFace transformers 的模型，首先需要下载预训练 checkpoint，比如 HuggingFace Hub 中的 Bloom 模型：`bigscience/bloom-7b1`，或是本地保存好的 SFT 模型；
然后使用如下代码，将 HuggingFace transformers 模型转化为 Megatron-LM 模型格式；在这个例子中，我们会将模型转换成 `TP (tensor_model_parallel_size)=8，PP (pipeline_model_parallel_size)=1` 的 checkpoint, 模型会存放在`MEGATRON_BLOOM_CKPT_PATH`中。

```bash
MEGATRON=path-to-megatron
cd $MEGATRON

bash examples/pai/tools/convert_transformers_megatron_bloom.sh \
$MEGATRON \
path-to-transformer-model \
path-to-megatron-model \
8 \
1 \
false
```

### 开启 SFT 训练

下面的脚本是一个 SFT 的训练样例。其中 `DATASET_PATH` 为 SFT 训练集路径，比如`$DATASET_ROOT/sft/train.jsonl`，在这个例子中，我们假设 tokenizer 存放的路径和模型 checkpoint 存放的路径相同。

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm-chatlearn
cd ${CHATLEARN}/examples/megatron/step1_sft/

LOAD_PATH=$MEGATRON_BLOOM_CKPT_PATH \
TOKENIZER_PATH=$MEGATRON_BLOOM_CKPT_PATH \
DATASET_PATH=$DATASET_ROOT/sft/ \
bash bloom_sft.sh
```

训练 log 和训练完成的模型默认会存放在`${CHATLEARN}/output/step1_sft`中，具体的定义详见`${CHATLEARN}/examples/megatron/step1_sft/bloom_sft.sh`脚本。

7B SFT 训练需要 8 A100-80GB/A800-80GB/H800-80GB GPU 卡的资源。分布式执行所需的环境变量和配置参考 [分布式执行](run.md)。

## Step2: Reward 模型训练

Reward 模型指的是在 RLHF 中作为人类评价的代理，对模型产生的问题回复进行实时评价打分的模型，Reward 模型输入问题以及模型回复，可以产生一个标量表示模型回复的质量。


### 开启 Reward 模型训练

依据 InstructGPT[1]，Reward 模型训练基于 SFT 训练产生的模型 checkpoint 初始化，训练代码如下：

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm-chatlearn
cd ${CHATLEARN}/examples/megatron/step2_reward/

LOAD_PATH=path-to-sft-ckpt \
TOKENIZER_PATH=$MEGATRON_BLOOM_CKPT_PATH \
DATASET_PATH=$DATASET_ROOT/rm/ \
bash bloom_reward.sh
```

训练 log 和训练完成的模型默认会存放在`${CHATLEARN}/output/step2_reward`中，具体的定义详见`${CHATLEARN}/examples/megatron/step2_reward/bloom_reward.sh`脚本。 
分布式执行所需的环境变量和配置参考 [分布式执行](run.md)。

## Step3: RLHF 训练
RLHF 指的是在一个只有指令的数据集上尝试不同的回复然后吸取 Reward 模型给不同回复的 reward 的监督信号的过程。

### 开启 RLHF 训练

[阿里云 PAI DLC](https://www.aliyun.com/activity/bigdata/pai-dlc)[2]可以非常便捷高效地支持 RLHF 任务的训练。以下是一个 Bloom-7B 的 Policy 和 7B 的 Reward 模型的训练脚本。在这个例子中，用户需要设置 `POLICY_LOAD` 为 SFT 产出的 checkpoint 路径，Policy 模型和 Reference 模型将以 SFT 的 checkpoint 初始化。`REWARD_LOAD` 为 Reward 训练产出的 checkpoint 路径，同时，用户可以指定 load checkpoint 对应的 iteration 数。Reward 模型和 Value 模型将以 Reward 模型的权重作初始化。`VOCAB_FILE` 为 `BloomTokenizer` 所需文件所在的文件夹路径。

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm-chatlearn
export DATASET_PATH=$DATASET_ROOT/rlhf/train.jsonl

cd ${CHATLEARN}/examples/megatron/step3_rlhf

export exp_name=any_experiment_name_you_like

POLICY_LOAD=path-to-sft-ckpt \
REWARD_LOAD=path-to-trained-rm-checkpoint \
REWARD_LOAD_ITERATION=1000 \
VOCAB_FILE=path-to-vocab-file \
bash run_scripts/bloom/run_7b1_7b1.sh
```

在我们的训练脚本里，7B Policy + 7B Reward 的 RLHF 训练资源需要 8 A100-80GB/A800-80GB/H800-80GB GPU 卡的资源。

分布式执行所需的环境变量和配置参考 [分布式执行](run.md)。
**注意对于 RLHF 任务，如果在 PAI DLC 上运行，您需要填写高级配置`customPortList=30000-30050,createSvcForAllWorkers=true`。**


### 效果评估
首先，我们可以通过 ChatLearn 的模型转换工具将 Megatron-LM 格式的模型转换为 HuggingFace transformers 模型格式。

```bash
MEGATRON=path-to-megatron-lm-chatlearn
cd $MEGATRON

bash examples/pai/tools/convert_transformers_megatron_bloom.sh \
$MEGATRON \
ckpt-to-rlhf-policy-ckpt \
path-to-transformers-ckpt-path \
1 \
1 \
true
```

我们在 MT-Bench 上使用 GPT-4 API 测评了 Bloom 在 HH 数据集上 SFT 后和 RLHF 后的效果，可以看到相比于 SFT 后的模型，RLHF 提升了模型的平均表现。且在 Extraction、Math、Reasoning、STEM、Writing 项上均有所提升。我们这里的性能提升来自于开源 HH 数据集训练的 Reward 模型，使用用户自己定制的 Reward 模型有助于取得更好的效果。

| Model | Coding | Extraction | Humanities | Math | Reasoning | Roleplay | STEM | Writing | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bloom_sft | 1.45 | 1.1 | 3.35 | 1.45 | 2.6 | 3.1 | 2.65 | 1.4 | 2.27 |
| bloom_rlhf | 1.4 | **1.4** | 3.05 | **1.5** | **2.65** | 3.05 | **3.05** | **1.6** | **2.35** |

## Reference

1. Training language models to follow instructions with human feedback，[https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

