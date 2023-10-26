# 基于 LLaMA2 模型的端到端训练教程

本文档介绍基于 ChatLearn, Megatron-LM 框架和 LLaMA/LLaMA2 模型的训练流程。包含三阶段的训练：SFT, Reward 和 RLHF 训练。


**以下是这个 Tutorial 脚本中使用的通用环境变量集合：**

| ENV | 含义                                                                                                                            |
| --- |-------------------------------------------------------------------------------------------------------------------------------|
| `CHATLEARN` | ChatLearn 代码仓库 clone 存放的位置 [https://github.com/alibaba/ChatLearn.git](https://github.com/alibaba/ChatLearn.git)               |
| `MEGATRON` | Megatron-LM 代码仓库 clone 存放的位置 [https://github.com/NVIDIA/Megatron-LM.git](https://github.com/NVIDIA/Megatron-LM.git) |
| `DATASET_ROOT` | 存放SFT/Reward/RLHF训练数据集合的根目录                                                                                                   |
| `TOKENIZER_MODEL` | Tokenizer 使用的 tokenizer_model 所在的路径                                                                                           |


# Setup: 镜像、代码、数据准备

## 镜像
推荐参考 `https://github.com/alibaba/ChatLearn/tree/master/docker/ngc/Dockerfile.ngc23.09` 准备镜像。

## 代码

在这个示例中，我们需要下载以下相关代码。

```bash
# 下载为支持Megatron-LM
git clone https://github.com/NVIDIA/Megatron-LM.git
git checkout 954a65b04
# 下载ChatLearn代码
git clone https://github.com/alibaba/ChatLearn.git
```

## 数据

请参考 [三阶段数据](data.md) 准备好您的训练数据。


# Step1: SFT

SFT 指的是使用有标注的对话数据来微调预训练语言模型的过程。在这个示例中，我们需要下载预训练的模型，然后开始一个简单的 SFT 训练示例。


## 1.1 下载和转化预训练模型

若使用来自于 HuggingFace transformers 的模型，首先需要下载预训练 checkpoint，比如 HuggingFace Hub 中的 LLaMA2 模型：`meta-llama/Llama-2-7b-hf`，或是本地保存好的 SFT 模型；
然后使用如下代码，将 HuggingFace transformers 模型转化为 Megatron-LM 模型格式；
1. 对于7B的模型，我们会将模型转换成 `TP (tensor_model_parallel_size)=4，PP (pipeline_model_parallel_size)=1` 的 checkpoint, 模型会存放在`MEGATRON_LLAMA_CKPT_PATH`中。
2. 对于13B的模型，我们会将模型转化成 `TP=8，PP=1` 的 checkpoint。
3. 对于70B的模型，我们会将模型转化成 `TP=8，PP=4` 的 checkpoint。

```bash
MEGATRON=path-to-megatron
cd $MEGATRON

HF_FORMAT_DIR=path-to-hf-model
TOKENIZER_MODEL=$HF_FORMAT_DIR/tokenizer.model
MEGATRON_FORMAT_DIR=path-to-meg-model

python tools/checkpoint/util.py \
    --model-type GPT \
    --loader llama2_hf \
    --saver megatron \
    --target-tensor-parallel-size $TP \
    --target-pipeline-parallel-size $PP \
    --load-dir ${HF_FORMAT_DIR} \
    --save-dir ${MEGATRON_FORMAT_DIR} \
    --tokenizer-model ${TOKENIZER_MODEL}
```


## 1.2 开启 SFT 训练

下面的脚本是一个 SFT 的训练样例。其中 `DATASET_PATH` 为 SFT 训练集路径，比如`$DATASET_ROOT/sft/train.jsonl`。
其中 `MODEL_SIZE` 为脚本中指定模型大小的环境变量，可以为 `7B`/`13B`/`70B`。

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm-extension
cd ${CHATLEARN}/examples/megatron/step1_sft/

MODEL_SIZE=$MODEL_SIZE \
LOAD_PATH=$MEGATRON_LLAMA2_CKPT_PATH \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
DATASET_PATH=$DATASET_ROOT/sft/ \
bash llama2_sft.sh
```

训练 log 和训练完成的模型默认会存放在`${CHATLEARN}/output/step1_sft`中，具体的定义详见`${CHATLEARN}/examples/megatron/step1_sft/llama2_sft.sh`脚本。

在我们的训练脚本里，资源需求 (假设资源为 A100-80GB/A800-80GB/H800-80GB GPU) 如下：
1. 7B SFT: 8 GPU
2. 13B SFT: 8 GPU
3. 70B SFT: 4*8 GPU

分布式执行所需的环境变量和配置参考 [分布式执行](run.md)。

# Step2: Reward 模型训练

Reward 模型指的是在 RLHF 中作为人类评价的代理，对模型产生的问题回复进行实时评价打分的模型，Reward 模型输入问题以及模型回复，可以产生一个标量表示模型回复的质量。

## 2.1 开启 Reward 模型训练

依据 InstructGPT[1]，Reward 模型训练基于 SFT 训练产生的模型 checkpoint 初始化，训练代码如下：

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm-extension
cd ${CHATLEARN}/examples/megatron/step2_reward/

LOAD_PATH=path-to-sft-ckpt \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
DATASET_PATH=$DATASET_ROOT/rm/ \
bash llama2_reward.sh
```

训练 log 和训练完成的模型默认会存放在`${CHATLEARN}/output/step2_reward`中，具体的定义详见`${CHATLEARN}/examples/megatron/step2_reward/llama2_reward.sh`脚本。

相同规模的 Reward 模型训练所需的资源需求和 SFT 是一样的。

分布式执行所需的环境变量和配置参考 [分布式执行](run.md)。

# Step3: RLHF 训练
RLHF 指的是在一个只有指令的数据集上尝试不同的回复然后吸取 Reward 模型给不同回复的 reward 的监督信号的过程。


## 3.1 开启 RLHF 训练

以下是一个 LLaMA2-7B 的 Policy 和 7B 的 Reward 模型的训练脚本。
在这个例子中，用户需要设置 `POLICY_LOAD` 为 SFT 产出的 checkpoint 路径，Policy 模型和 Reference 模型将以 SFT 的 checkpoint 初始化。
`REWARD_LOAD` 为 Reward 训练产出的 checkpoint 路径，同时，用户可以指定 load checkpoint 对应的 iteration 数。
Reward 模型和 Value 模型将以 Reward 模型的权重作初始化。`TOKENIZER_MODEL` 为 `LlamaTokenizer` 所需文件 `tokenizer.model` 所在的文件夹路径。

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm
export DATASET_PATH=$DATASET_ROOT/rlhf/train.jsonl

cd ${CHATLEARN}/examples/megatron/step3_rlhf

export exp_name=any_experiment_name_you_like

POLICY_LOAD=path-to-sft-ckpt \
REWARD_LOAD=path-to-trained-rm-checkpoint \
REWARD_LOAD_ITERATION=1000 \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
bash run_scripts/llama2/run_7b_7b.sh
```

如果您需要训练 13B / 70B 的模型，只需要将上述训练脚本中的 `run_7b_7b.sh` 替换成 `run_13b_13b.sh` / `run_70b_70b.sh`。
您也可以根据自己的需求修改模型配置和其他参数。

在我们的训练脚本里，资源需求 (假设资源为 A100-80GB/A800-80GB/H800-80GB GPU) 如下：
1. 7B RLHF: 8 GPU
2. 13B RLHF: 2*8 GPU
3. 70B RLHF: 4*8 GPU

分布式执行所需的环境变量和配置参考 [分布式执行](run.md)。
**注意对于 RLHF 任务，如果在 PAI DLC 上运行，您需要填写高级配置`customPortList=30000-30050,createSvcForAllWorkers=true`。**


## 3.2 效果评估

首先，我们可以通过 ChatLearn 的模型转换工具将 Megatron-LM 格式的模型转换为 HuggingFace transformers 模型格式。

```bash
cd $CHATLEARN
python chatlearn/tools/megatron_to_hf.py \
  --load_path ${dir-to-megatron-model} \
  --save_path ${save-dir} \
  --target_params_dtype bf16 \
  --vocab_dir ${dir-of-vocab-file} \
  --megatron_path ${dir-to-megatron}
```

- `load_path` 为需要转化的Megatron checkpoint所在的文件夹，要求 checkpoint 并行策略为 `TP=1, PP=1`。
- `save_dir` 为转化后的 HF Transformer 模型所在的文件夹。
- `target_params_dtype` 为转化模型的数据类型。
- `vocab_dir` 为 `tokenizer.model` 等文件所在的文件夹。
- `megatron_path` 为 Megatron-LM 所在的文件夹。

我们在 MT-Bench 上使用 GPT-4 API 测评了 LLaMA-13B 在 HH 数据集上 SFT 后和 RLHF 后的效果，可以看到相比于 SFT 后的模型，RLHF 提升了模型的平均表现。
且在 Humanities、Math、Roleplay、STEM、Writing 项上有显著的提升。我们这里的性能提升来自于开源 HH 数据集训练的 Reward 模型，使用用户自己定制的 Reward 模型有助于取得更好的效果。

| Model | Coding | Extraction | Humanities | Math | Reasoning | Roleplay | STEM | Writing | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_sft | 1.6 | 2.7 | 4.2 | 1.1 | 2.85 | 3.35 | 4.55 | 2.95 | 2.90 |
| llama_rlhf | **1.75** | **3.45** | **4.75** | **1.55** | **3.5** | **5.85** | **5.0** | **5.0** | **3.85** |

# Reference

1. Training language models to follow instructions with human feedback，[https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)
