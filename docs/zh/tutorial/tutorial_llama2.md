# 基于 Llama 模型的端到端训练教程

本文档介绍基于 ChatLearn, Megatron-LM 和 vLLM 框架和 Llama/Llama2 模型进行 alignment 的训练流程。支持RLHF、DPO、OnlineDPO、GRPO 多种训练模式：
1. RLHF(Reinforcement Learning from Human Feedback)：包括三阶段的训练（SFT, Reward 和 RLHF 训练）;
2. DPO(Direct Preference Optimization)：包括两阶段的训练（SFT 和 DPO 训练）;
3. OnlineDPO/GRPO：介于 DPO 和 RLHF 之间，使用 Policy + Reward 模型来自动生成数据并进行打分，再进行DPO训练，包括三阶段的训练（SFT, Reward 和 DPO 训练）.


**以下是这个 Tutorial 脚本中使用的通用环境变量集合：**

| ENV | 含义                                                                                                                            |
| --- |-------------------------------------------------------------------------------------------------------------------------------|
| `CHATLEARN` | ChatLearn 代码仓库 clone 存放的位置 [https://github.com/alibaba/ChatLearn.git](https://github.com/alibaba/ChatLearn.git)               |
| `MEGATRON` | Megatron-LM 代码仓库 clone 存放的位置 [https://github.com/NVIDIA/Megatron-LM.git](https://github.com/NVIDIA/Megatron-LM.git) |
| `DATASET_ROOT` | 存放SFT/Reward/RLHF/DPO/OnlineDPO/GRPO训练数据集合的根目录                                                                                                   |
| `TOKENIZER_MODEL` | Tokenizer 使用的 tokenizer_model 所在的路径                                                                                           |


## Setup: 镜像、代码、数据准备

### 镜像和代码

请参考 [镜像和代码准备](../installation.md)。

### 数据

请参考 [各阶段数据](data.md) 准备好您的训练数据。


## SFT

SFT 指的是使用有标注的对话数据来微调预训练语言模型的过程。在这个示例中，我们需要下载预训练的模型，然后开始一个简单的 SFT 训练示例。


### 下载和转化预训练模型

若使用来自于 HuggingFace transformers 的模型，首先需要下载预训练 checkpoint，比如 HuggingFace Hub 中的 Llama2 模型：`meta-llama/Llama-2-7b-hf`，或是本地保存好的 SFT 模型；
然后使用如下代码，将 HuggingFace transformers 模型转化为 Megatron-LM 的 Legacy 模型格式；
1. 对于llama2-7B的模型，我们会将模型转换成 `TP (tensor_model_parallel_size)=4，PP (pipeline_model_parallel_size)=1` 的 checkpoint, 模型会存放在`SAVE_PATH`中。
2. 对于llama2-13B的模型，我们会将模型转化成 `TP=8，PP=1` 的 checkpoint。
3. 对于llama2-70B的模型，我们会将模型转化成 `TP=8，PP=4` 的 checkpoint。

```bash
export MEGATRON=path-to-megatron-lm
export CHATLEARN=path-to-chatlearn

cd ${CHATLEARN}/examples/megatron/

TP=num_of_tp \
PP=num_of_pp \
LOAD_PATH=path-to-hf-model \
TOKENIZER_MODEL=$LOAD_PATH/tokenizer.model \
SAVE_PATH=path-to-megatron-model \
bash scripts/convert_hf_to_megatron.sh
```


### 开启 SFT 训练

下面的脚本是一个 SFT 的训练样例。其中 `DATASET_PATH` 为 SFT 训练集路径，比如`$DATASET_ROOT/sft/train.jsonl`。
其中 `model_size` 为脚本中指定模型大小的环境变量，可以为 `llama2-7B`/`llama2-13B`/`llama2-70B`。

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm
cd ${CHATLEARN}/examples/megatron/

export model_size=llama2-7B

LOAD_PATH=$MEGATRON_LLAMA2_CKPT_PATH \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
DATASET_PATH=$DATASET_ROOT/sft/ \
bash scripts/train_sft_llama.sh
```

训练 log 和训练完成的模型默认会存放在`${CHATLEARN}/output/sft`中，可以通过 CHECKPOINT_PATH 来指定模型保存路径，具体的定义详见`${CHATLEARN}/examples/megatron/scripts/train_sft_llama.sh`脚本。

在我们的训练脚本里，资源需求 (假设资源为 A100-80GB/A800-80GB GPU) 如下：
1. llama2-7B SFT: 8 GPU
2. llama2-13B SFT: 8 GPU
3. llama2-70B SFT: 4*8 GPU

分布式执行所需的环境变量和配置参考 [分布式执行](run.md)。

## Reward 模型训练

Reward 模型指的是在 RLHF 中作为人类评价的代理，对模型产生的问题回复进行实时评价打分的模型，Reward 模型输入问题以及模型回复，可以产生一个标量表示模型回复的质量。

**注**：DPO训练模式不需要训练Reward模型。

### 开启 Reward 模型训练

依据 InstructGPT[1]，Reward 模型训练基于 SFT 训练产生的模型 checkpoint 初始化，训练代码如下：

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm
cd ${CHATLEARN}/examples/megatron/

LOAD_PATH=path-to-sft-ckpt \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
DATASET_PATH=$DATASET_ROOT/rm/ \
bash scripts/train_reward_llama.sh
```

训练 log 和训练完成的模型默认会存放在`${CHATLEARN}/output/reward`中，具体的定义详见`${CHATLEARN}/examples/megatron/scripts/train_reward_llama.sh`脚本。

相同规模的 Reward 模型训练所需的资源需求和 SFT 是一样的。

分布式执行所需的环境变量和配置参考 [分布式执行](run.md)。

## Alignment 训练

ChatLearn 支持多种 Alignment 训练模式：RLHF、DPO、OnlineDPO、GRP、GRPO


### 开启 Alignment 训练

以下是一个Llama2-7B规模模型训练的使用范例。

#### RLHF

以下是一个 Llama2-7B 的 Policy 和 7B 的 Reward 模型的训练脚本。
在这个例子中，用户需要设置 `POLICY_LOAD` 为 SFT 产出的 checkpoint 路径，Policy 模型和 Reference 模型将以 SFT 的 checkpoint 初始化。
`REWARD_LOAD` 为 Reward 训练产出的 checkpoint 路径，同时，用户可以指定 load checkpoint 对应的 iteration 数。
Reward 模型和 Value 模型将以 Reward 模型的权重作初始化。`TOKENIZER_MODEL` 为 `Llama2Tokenizer` 所需文件 `tokenizer.model` 所在的文件夹路径。

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm
export DATASET_PATH=$DATASET_ROOT/alignment/train.jsonl

cd ${CHATLEARN}/examples/megatron/

export model_size=llama2-7B

POLICY_LOAD=path-to-sft-ckpt \
REWARD_LOAD=path-to-rm-ckpt \
REWARD_LOAD_ITERATION=1000 \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
bash scripts/train_rlhf_llama.sh
```

#### OnlineDPO/GRPO

OnlineDPO/GRPO训练流程和RLHF比较类似，只是不需要Value模型，以下是一个 Llama2-7B 的 Policy 和 7B 的 Reward 模型的训练脚本。

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm
export DATASET_PATH=$DATASET_ROOT/alignment/train.jsonl

cd ${CHATLEARN}/examples/megatron/

export model_size=llama2-7B

POLICY_LOAD=path-to-sft-ckpt \
REWARD_LOAD=path-to-rm-ckpt \
REWARD_LOAD_ITERATION=1000 \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
bash scripts/train_online_dpo_llama.sh
```

#### DPO
以下是一个 Llama2-7B 的 Policy模型的训练脚本。
在这个例子中，用户需要设置 `POLICY_LOAD` 为 SFT 产出的 checkpoint 路径，Policy 模型和 Reference 模型将以 SFT 的 checkpoint 初始化。
`TOKENIZER_MODEL` 为 `Llama2Tokenizer` 所需文件 `tokenizer.model` 所在的文件夹路径。

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm
export DATASET_PATH=$DATASET_ROOT/alignment/train.jsonl

cd ${CHATLEARN}/examples/megatron/

export model_size=llama2-7B

POLICY_LOAD=path-to-sft-ckpt \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
bash scripts/train_dpo_llama.sh
```

#### GRPO Math

如果用户需要训练一个 GRPO Math 模型，需要先参考 [Math data](data.md#Math) 准备好数学数据集。以下为一个 Llama2-7B 的模型训练范例。

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm
export DATASET_PATH=$DATASET_ROOT/math/train.jsonl

cd ${CHATLEARN}/examples/megatron/

export model_size=llama2-7B

POLICY_LOAD=path-to-sft-ckpt \
REWARD_LOAD=path-to-rm-ckpt \
REWARD_LOAD_ITERATION=1000 \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
bash scripts/train_grpo_math_llama.sh
```


### 更大规模参数模型范例

如果您需要训练 llama2-13B / llama2-70B 的模型，只需要将上述训练脚本中的 `export model_size=llama2-7B` 替换成 `export model_size=llama2-13B` / `export model_size=llama2-70B`。
您也可以根据自己的需求修改模型配置和其他参数。

在我们的训练脚本里，资源需求 (假设资源为 A100-80GB/A800-80GB GPU) 如下：
1. llama2-7B RLHF: 8 GPU
2. llama2-13B RLHF: 2*8 GPU
3. llama2-70B RLHF: 4*8 GPU

分布式执行所需的环境变量和配置参考 [分布式执行](run.md)。


### 效果评估

首先，我们可以通过 ChatLearn 的模型转换工具将 Megatron-LM 的 Legacy 格式的模型转换为 HuggingFace transformers 模型格式。

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm

cd $CHATLEARN/examples/megatron/

LOAD_PATH=path-to-megatron-model \
SAVE_PATH=path-to-hf-model \
VOCAB_PATH=path-to-vocab \
target_params_dtype=bf16 \
bash scripts/convert_megatron_to_hf.sh
```

- `load_path` 为需要转化的Megatron checkpoint所在的文件夹。
- `save_path` 为转化后的 HF Transformer 模型所在的文件夹。
- `target_params_dtype` 为转化模型的数据类型。
- `vocab_path` 为 `tokenizer.model` 等文件所在的文件夹。

我们在 MT-Bench 上使用 GPT-4 API 测评了 Llama2-7B 在 HH 数据集上的 SFT、RLHF、DPO 和 OnlineDPO 后的效果，可以看到相比于 SFT 后的模型，使用 ChatLearn 提供的 RLHF、DPO 和 OnlineDPO 等对齐训练方法提升了模型的平均表现。其中，RLHF 在 Humanities、Math、Roleplay、Reasoning、Writing 项上有显著的提升。我们这里的性能提升来自于开源 HH 数据集训练的 Reward 模型，使用用户自己定制的 Reward 模型有助于取得更好的效果。


| Metric      | llama_sft | llama_rlhf | llama_dpo | llama_onlinedpo |
|-------------|-----------|------------|-----------|------------------|
| Coding      | 2.05      | **1.65**   | **2.17**  | **1.75**         |
| Extraction  | 4.40      | **4.0**    | **4.35**  | **3.70**         |
| Humanities  | 5.85      | **7.17**   | **6.70**  | **7.52**         |
| Math        | 1.15      | **1.70**   | **1.25**  | **1.05**         |
| Reasoning   | 3.15      | **3.30**   | **3.15**  | **2.00**         |
| Roleplay    | 4.75      | **5.50**   | **5.65**  | **6.10**         |
| STEM        | 6.05      | **5.75**   | **6.77**  | **7.10**         |
| Writing     | 4.55      | **4.75**   | **4.8**   | **5.30**         |
| Avg         | 3.94      | **4.22**   | **4.33**  | **4.31**         |


### 使用 Megatron-Core 模型格式

如果您需要训练 Megatron-Core 格式的模型，您只需要为上面每一个脚本添加 `USE_LEGACY_MODELS=False` 参数。这个参数的作用是控制 Megatron-LM 是否选择 Megatron-Core 模型格式进行训练。它的默认值是 `True` ，代表 Megatron-LM 默认选择 Legacy 模型格式。若设定它的值为 `False` ，则代表 Megatron-LM 选择 Megatron-Core 模型格式。

#### 使用范例

以使用 RLHF 训练模式训练一个 Llama-7B 模型为例，SFT、Reward 和 RLHF 训练步骤如下：

1. 将 HuggingFace transformers 模型转化为 Megatron-Core 模型格式

```bash
export MEGATRON=path-to-megatron-lm
export CHATLEARN=path-to-chatlearn

cd ${CHATLEARN}/examples/megatron/

TP=num_of_tp \
PP=num_of_pp \
LOAD_PATH=path-to-hf-model \
TOKENIZER_MODEL=$LOAD_PATH/tokenizer.model \
SAVE_PATH=path-to-megatron-model \
USE_LEGACY_MODELS=False \
bash scripts/convert_hf_to_megatron.sh
```

2. 开启 SFT 训练

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm
cd ${CHATLEARN}/examples/megatron/

export model_size=llama2-7B

LOAD_PATH=$MEGATRON_LLAMA2_CKPT_PATH \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
DATASET_PATH=$DATASET_ROOT/sft/ \
USE_LEGACY_MODELS=False \
bash scripts/train_sft_llama.sh
```

3. 开启 Reward 训练

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm
cd ${CHATLEARN}/examples/megatron/

LOAD_PATH=path-to-sft-ckpt \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
DATASET_PATH=$DATASET_ROOT/rm/ \
USE_LEGACY_MODELS=False \
bash scripts/train_reward_llama.sh
```

4. 开启 RLHF 训练

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm
export DATASET_PATH=$DATASET_ROOT/alignment/train.jsonl

cd ${CHATLEARN}/examples/megatron/

export model_size=llama2-7B

POLICY_LOAD=path-to-sft-ckpt \
REWARD_LOAD=path-to-rm-ckpt \
REWARD_LOAD_ITERATION=1000 \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
USE_LEGACY_MODELS=False \
bash scripts/train_rlhf_llama.sh
```

#### 效果评估

首先通过 ChatLearn 的模型转换工具将 Megatron-Core 格式的模型转换为 HuggingFace transformers 模型格式。

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm

cd $CHATLEARN/examples/megatron/

LOAD_PATH=path-to-megatron-model \
SAVE_PATH=path-to-hf-model \
VOCAB_PATH=path-to-vocab \
target_params_dtype=bf16 \
USE_LEGACY_MODELS=False \
bash scripts/convert_megatron_to_hf.sh
```

我们在 MT-Bench 上使用 GPT-4 API 测评了 Llama2-7B 在 HH 数据集上使用 Megatron-Core 模型格式的 SFT 后和 RLHF 后的效果，可以看到相比于 SFT 后的模型，RLHF 提升了模型的平均表现。其中，RLHF 在 Humanities、Roleplay、STEM、Writing 项上有显著的提升。我们这里的性能提升来自于开源 HH 数据集训练的 Reward 模型，使用用户自己定制的 Reward 模型有助于取得更好的效果。


| Metric      | llama_sft | llama_rlhf |
|-------------|-----------|------------|
| Coding      | 1.95      | **1.45**   |
| Extraction  | 3.80      | **3.95**   |
| Humanities  | 6.45      | **7.10**   |
| Math        | 1.80      | **1.75**   |
| Reasoning   | 3.60      | **2.75**   |
| Roleplay    | 4.60      | **5.40**   |
| STEM        | 5.25      | **7.15**   |
| Writing     | 4.35      | **4.95**   |
| Avg         | 3.98      | **4.31**   |


## Reference

1. Training language models to follow instructions with human feedback，[https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

