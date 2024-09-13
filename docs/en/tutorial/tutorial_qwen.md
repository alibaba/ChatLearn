# End-to-end training tutorial based on the Qwen model

This document describes DPO training based on the ChatLearn, DeepSpeed framework, and Qwen model.

**The following is a collection of common environment variables used in this tutorial script:**
| ENV | Meaning                                                                                                                       |
| --- |-------------------------------------------------------------------------------------------------------------------------------|
| `CHATLEARN` | Location where the ChatLearn code repository is cloned [https://github.com/alibaba/ChatLearn.git](https://github.com/alibaba/ChatLearn.git)                |
| `DATASET_ROOT` | Root directory where the training datasets are stored                                                                        |

## Setup: Image, Code, and Data Preparation

### Image / Code

Please refer to [Environment and Code Setup](../installation.md).

### Data
The data format required by qwen2 is chatml:
```
{"type": "chatml", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Tell me something about large language models."}, {"role": "assistant", "content": "Large language models are a type of language model that is trained on a large corpus of text data. They are capable of generating human-like text and are used in a variety of natural language processing tasks..."}], "source": "unknown"}
```
The following script can convert `Dahoas/full-hh-rlhf` to data in chatml format and store it in the file `$DATASET_ROOT/alignment/train.jsonl`:
```bash
cd ${CHATLEARN}/examples/huggingface/
DATASET_ROOT=path-to-dataset-root
python data/preprocess_data_chatml.py $DATASET_ROOT
```

## DPO
Here is an example of DPO training for Qwen2-7B.
In this example, the user needs to set `policy_model_path` to the initialization model checkpoint path, and the Policy model and Reference model will be initialized with this checkpoint.
```
export CHATLEARN=path-to-chatlearn
export DATASET_PATH=$DATASET_ROOT/alignment/train.jsonl
export policy_model_path=path-to-qwen2-ckpt
cd ${CHATLEARN}/examples/huggingface/
bash scripts/train_dpo_qwen.sh
```
