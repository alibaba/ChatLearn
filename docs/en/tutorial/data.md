# Data

This document describes the data preparation process for three stages: SFT, Reward, and RLHF.


**The following is a collection of general environment variables used in this tutorial script:**

| ENV | Explanation |
| --- | --- |
| `CHATLEARN` | The location where the ChatLearn code is cloned [https://github.com/alibaba/ChatLearn.git](https://github.com/alibaba/ChatLearn.git) |
| `DATASET_ROOT` | The root directory for storing the SFT/Reward/RLHF training dataset collection. |



## 1.1 Prepare SFT Training Data

Organize the question-response pairs of SFT data into a jsonl file, where each line of the jsonl file represents a SFT data sample in the following Python dictionary format:

```json
{'query': question, 'response': reply}
```

Taking the example of Anthropic's helpful&harmless data, use the following code to store it in `$DATASET_ROOT/sft/train.jsonl`.

```bash
cd ${CHATLEARN}/examples/megatron/step1_sft/
DATASET_ROOT=$path_to_dataset_root
python prepare_data.py $DATASET_ROOT
```

## 2.1 Prepare Reward Training Data

1. First, prepare question-different response pairs and organize them into a jsonl file. Each line in the jsonl file represents a Reward model training data sample in the following Python dictionary format:

```json
{'query': question, 'response': [reply 1, reply 2, ...], 'score': [score1, score2, ...]}
```

The score value indicates the quality of the corresponding response, with higher scores indicating higher quality and closer to human preference.

2. Taking the example of Anthropic's helpful&harmless data, use the following code to store it in `$DATASET_ROOT/rm/train.jsonl` and `$DATASET_ROOT/rm/dev.jsonl`.

```bash
cd ${CHATLEARN}/examples/megatron/step2_reward/
DATASET_ROOT=path-to-dataset-root
python prepare_data.py $DATASET_ROOT
```

## 3.1 Prepare RLHF Training Data

1. Firstly, prepare a dataset of instructions to be explored and organize it into a JSON file. Each line in the JSON file should represent a prompt in the following format:

```json
{"prompt": prompt}
```

2. Taking Anthropic's helpful & harmless data as an example, use the following code to store the dataset in `$DATASET_ROOT/rlhf/train.jsonl` and `$DATASET_ROOT/rlhf/dev.jsonl`:

```bash
cd ${CHATLEARN}/examples/megatron/step3_rlhf/
DATASET_ROOT=path-to-dataset-root
python prepare_data.py $DATASET_ROOT
```