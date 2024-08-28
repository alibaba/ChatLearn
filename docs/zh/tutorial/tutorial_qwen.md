# 基于 Qwen 模型的端到端训练教程

本文档介绍基于 ChatLearn, DeepSpeed 框架和 Qwen 模型进行 DPO 训练。

**以下是这个 Tutorial 脚本中使用的通用环境变量集合：**

| ENV | 含义                                                                                                                            |
| --- |-------------------------------------------------------------------------------------------------------------------------------|
| `CHATLEARN` | ChatLearn 代码仓库 clone 存放的位置 [https://github.com/alibaba/ChatLearn.git](https://github.com/alibaba/ChatLearn.git)               |
| `DATASET_ROOT` | 存放训练数据集合的根目录                                                                                                   |


## Setup: 镜像、代码、数据准备

### 镜像和代码

请参考 [镜像和代码准备](../installation.md)。

### 数据

qwen2 要求的数据格式为chatml

```
{"type": "chatml", "messages": [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": "Tell me something about large language models."}, {"role": "assistant", "content": "Large language models are a type of language model that is trained on a large corpus of text data. They are capable of generating human-like text and are used in a variety of natural language processing tasks..."}], "source": "unknown"}
```
通过以下脚本可以将 `Dahoas/full-hh-rlhf` 转换为 chatml 格式的数据, 并存储 `$DATASET_ROOT/alignment/train.jsonl` 文件.

```bash
cd ${CHATLEARN}/examples/huggingface/
DATASET_ROOT=path-to-dataset-root
python data/preprocess_data_chatml.py $DATASET_ROOT
```


### DPO

以下是一个 Qwen2-7B 的 DPO 训练范例。
在这个例子中，用户需要设置 `policy_model_path` 为 初始化模型 checkpoint 路径，Policy 模型和 Reference 模型将以这个 checkpoint 初始化。

```
export CHATLEARN=path-to-chatlearn
export DATASET_PATH=$DATASET_ROOT/alignment/train.jsonl
export policy_model_path=path-to-qwen2-ckpt
cd ${CHATLEARN}/examples/huggingface/
bash scripts/train_dpo_qwen.sh
```
