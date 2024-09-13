# Data

This document describes the data preparation process for different stages: SFT, Reward, RLHF, DPO, OnlineDPO and GRPO.


**The following is a collection of general environment variables used in this tutorial script:**

| ENV | Explanation |
| --- | --- |
| `CHATLEARN` | The location where the ChatLearn code is cloned [https://github.com/alibaba/ChatLearn.git](https://github.com/alibaba/ChatLearn.git) |
| `DATASET_ROOT` | The root directory for storing the SFT/Reward/RLHF/DPO/OnlineDPO/GRPO training dataset collection. |



## 1 Prepare SFT Training Data

Organize the question-response pairs of SFT data into a jsonl file, where each line of the jsonl file represents a SFT data sample in the following Python dictionary format:

```
{'query': question, 'response': reply}
```

Taking the example of Anthropic's helpful&harmless data, use the following code to store it in `$DATASET_ROOT/sft/train.jsonl`.

```bash
cd ${CHATLEARN}/examples/megatron/
DATASET_ROOT=$path_to_dataset_root
python data/prepare_data_sft.py $DATASET_ROOT
```

## 2 Prepare Reward Training Data

1. First, prepare question-different response pairs and organize them into a jsonl file. Each line in the jsonl file represents a Reward model training data sample in the following Python dictionary format:

```
{'query': question, 'response': [reply 1, reply 2, ...], 'score': [score1, score2, ...]}
```

The score value indicates the quality of the corresponding response, with higher scores indicating higher quality and closer to human preference.

2. Taking the example of Anthropic's helpful&harmless data, use the following code to store it in `$DATASET_ROOT/rm/train.jsonl` and `$DATASET_ROOT/rm/dev.jsonl`.

```bash
cd ${CHATLEARN}/examples/megatron/
DATASET_ROOT=path-to-dataset-root
python data/prepare_data_reward.py $DATASET_ROOT
```

## 3 Prepare Alignment Training Data

ChatLearn supports multiple alignments: RLHF, DPO, OnlineDPO, GRPO

1. Firstly, prepare a dataset of instructions to be explored and organize it into a JSON file. Each line in the JSON file should represent a prompt in the following format:

```
{"prompt": prompt}
```

2. Taking Anthropic's helpful & harmless data as an example, use the following code to store the dataset in `$DATASET_ROOT/alignment/train.jsonl` and `$DATASET_ROOT/alignment/dev.jsonl`:

```bash
cd ${CHATLEARN}/examples/megatron/
DATASET_ROOT=path-to-dataset-root
python data/prepare_data_alignment.py $DATASET_ROOT
```
## 4 Prepare Math Training Data

1. Firstly, prepare a dataset of math data to be explored and organize it into a JSON file. Each line in the JSON file should represent a prompt in the following format:

```
{"eval_func": "math_rule", "prompt": prompt, 'answer': answer}
```

2. Taking openai/gsm8k data as an example, use the following code to store the dataset in `$DATASET_ROOT/math/train.jsonl`:

```bash
cd ${CHATLEARN}/examples/megatron/
DATASET_ROOT=path-to-dataset-root
python data/prepare_data_math.py $DATASET_ROOT
```