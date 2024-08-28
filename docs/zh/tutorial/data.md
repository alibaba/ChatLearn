# 数据准备

本文档介绍不同阶段 SFT, Reward，RLHF，DPO, OnlineDPO 和 GRPO 的数据准备流程。

**以下是这个 Tutorial 脚本中使用的通用环境变量集合：**

| ENV | 含义 |
| --- | --- |
| `CHATLEARN` | ChatLearn 代码仓库 clone 存放的位置 [https://github.com/alibaba/ChatLearn.git](https://github.com/alibaba/ChatLearn.git) |
| `DATASET_ROOT` | 存放SFT/Reward/RLHF/DPO/OnlineDPO/GRPO训练数据集合的根目录 |

## 准备 SFT 训练数据

将 SFT 数据的问题 - 回复配对的样本，整理到一个 jsonl 文件中，其中 jsonl 文件中每一行为一条 SFT 数据，形式为如下的 Python 字典格式：

```
{'query': 问题，'response': 回复}
```

以 Anthropic 的 helpful&harmless 的数据为例，使用如下代码，会存一个 `$DATASET_ROOT/sft/train.jsonl`.

```bash
cd ${CHATLEARN}/examples/megatron/
DATASET_ROOT=$path_to_dataset_root
python data/prepare_data_sft.py $DATASET_ROOT
```

## 准备 Reward 训练数据

1. 首先准备问题 - 不同回复配对的样本，整理到一个 jsonl 文件中，其中 jsonl 文件中每一行为一条 Reward 模型训练数据，形式为如下的 Python 字典格式：

```
{'query': 问题，'response': [回复 1, 回复 2, .....], 'score': [score1, score2, .....]}
```

其中 score 的值越高意味着对应回复的质量越高，越贴近人类偏好。

2. 以 Anthropic 的 helpful&harmless 的数据为例，使用如下代码，会存一个 `$DATASET_ROOT/rm/train.jsonl` 和 `$DATASET_ROOT/rm/dev.jsonl`.

```bash
cd ${CHATLEARN}/examples/megatron/
DATASET_ROOT=path-to-dataset-root
python data/prepare_data_reward.py $DATASET_ROOT
```

## 准备 Alignment 训练数据

ChatLearn中支持多种Alignment的训练模式：RLHF, DPO, OnlineDPO, GRPO

其中RLHF/OnlineDPO/GRPO数据格式相同。


### RLHF/OnlineDPO/GRPO

1. 首先准备一个需要被探索的指令数据集，整理到一个 jsonl 文件中，其中 jsonl 文件中每一行为一条指令，格式为

```
{"prompt": 问题}
```

2. 以 Anthropic 的 helpful&harmless 的数据为例，使用如下代码，会存一个`$DATASET_ROOT/alignment/train.jsonl` 和`$DATASET_ROOT/alignment/dev.jsonl`：

```bash
cd ${CHATLEARN}/examples/megatron/
DATASET_ROOT=path-to-dataset-root
python data/prepare_data_alignment.py $DATASET_ROOT
```

### DPO

准备一个需要被探索的指令数据集，整理到一个 jsonl 文件中，其中 jsonl 文件中每一行为一条指令，格式为：prompt+chosen+rejected，例如：

```
{"prompt": 问题, "chosen": 正偏好回答, "rejected": 负偏好回答}
```

其中prompt字段内容分为两种场景：
1. 单轮对话：仅包括单轮对话的`问题`；
2. 多轮对话：包含前几轮对话的问答及最后一轮的提问。

开源Anthropic的helpful&harmless的数据满足DPO训练需求，可参考RLHF章节下载相应训练数据。

### Math

首先，准备一个要训练的数学数据集，并将其组织成一个 JSON 文件。JSON 文件中的每一行应该表示一个样本，格式如下：

```
{"eval_func": "math_rule", "prompt": prompt, "answer": answer}
```

以 `openai/gsm8k` 数据为例，使用以下代码将数据集存储在 `$DATASET_ROOT/math/train.jsonl` 中：

```
cd ${CHATLEARN}/examples/megatron/
DATASET_ROOT=path-to-dataset-root
python data/prepare_data_math.py $DATASET_ROOT
```

