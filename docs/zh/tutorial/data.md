# 数据准备

本文档介绍三阶段 SFT, Reward 和 RLHF 的数据准备流程。

**以下是这个 Tutorial 脚本中使用的通用环境变量集合：**

| ENV | 含义 |
| --- | --- |
| `CHATLEARN` | ChatLearn 代码仓库 clone 存放的位置 [https://github.com/alibaba/ChatLearn.git](https://github.com/alibaba/ChatLearn.git) |
| `DATASET_ROOT` | 存放SFT/Reward/RLHF训练数据集合的根目录 |

## 准备 SFT 训练数据

将 SFT 数据的问题 - 回复配对的样本，整理到一个 jsonl 文件中，其中 jsonl 文件中每一行为一条 SFT 数据，形式为如下的 Python 字典格式：

```json
{'query': 问题，'response': 回复}
```

以 Anthropic 的 helpful&harmless 的数据为例，使用如下代码，会存一个 `$DATASET_ROOT/sft/train.jsonl`.

```bash
cd ${CHATLEARN}/examples/megatron/step1_sft/
DATASET_ROOT=$path_to_dataset_root
python prepare_data.py $DATASET_ROOT
```

## 准备 Reward 训练数据

1. 首先准备问题 - 不同回复配对的样本，整理到一个 jsonl 文件中，其中 jsonl 文件中每一行为一条 Reward 模型训练数据，形式为如下的 Python 字典格式：

```json
{'query': 问题，'response': [回复 1, 回复 2, .....], 'score': [score1, score2, .....]}
```

其中 score 的值越高意味着对应回复的质量越高，越贴近人类偏好。

2. 以 Anthropic 的 helpful&harmless 的数据为例，使用如下代码，会存一个 `$DATASET_ROOT/rm/train.jsonl` 和 `$DATASET_ROOT/rm/dev.jsonl`.

```bash
cd ${CHATLEARN}/examples/megatron/step2_reward/
DATASET_ROOT=path-to-dataset-root
python prepare_data.py $DATASET_ROOT
```

## 准备 RLHF 训练数据

1. 首先准备一个需要被探索的指令数据集，整理到一个 jsonl 文件中，其中 jsonl 文件中每一行为一条指令，格式为

```json
{"prompt": 问题}
```

2. 以 Anthropic 的 helpful&harmless 的数据为例，使用如下代码，会存一个`$DATASET_ROOT/rlhf/train.jsonl` 和`$DATASET_ROOT/rlhf/dev.jsonl`：

```bash
cd ${CHATLEARN}/examples/megatron/step3_rlhf/
DATASET_ROOT=path-to-dataset-root
python prepare_data.py $DATASET_ROOT
```
