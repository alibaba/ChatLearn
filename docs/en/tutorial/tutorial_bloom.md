# End-to-end Training Tutorial with Bloom Model

This document provides instructions for end-to-end training using the ChatLearn, Megatron-LM framework, and the Bloom model. It includes three stages of training: SFT, Reward, and RLHF training.


**The following is a collection of general environment variables used in this tutorial script:**

| ENV | Explanation |
| --- | --- |
| `CHATLEARN` | The location where the ChatLearn code is cloned [https://github.com/alibaba/ChatLearn.git](https://github.com/alibaba/ChatLearn.git) |
| `MEGATRON` | The location where the Megatron-LM-ChatLearn code is cloned [https://github.com/alibaba/Megatron-LM-ChatLearn.git](https://github.com/alibaba/Megatron-LM-ChatLearn.git) |
| `DATASET_ROOT` | The root directory for storing the SFT/Reward/RLHF training dataset collection. |
| `TOKENIZER_PATH` | The folder where the vocab_file used by the Tokenizer is located. |


## Setup: Image / Code and Data Preparation

### Docker Image
It is recommended to refer to `https://github.com/alibaba/ChatLearn/tree/master/docker/ngc/Dockerfile.ngc22.10` for preparing the docker image.
If you're training on the PAI DLC platform, we suggest using the pre-built image provided below:

```bash
registry.cn-wulanchabu.aliyuncs.com/pai-dlc/pytorch-training:1.13.0-gpu-py3.8-cu11.8-ngc22.10-ubuntu20.04
```

### Code

In this example, we need to download the following related code.

```bash
# Download the modified Megatron-LM-ChatLearn to support ChatLearn training
git clone -b v0.1.0 https://github.com/alibaba/Megatron-LM-ChatLearn.git
# Download the ChatLearn code
git clone -b v0.1.0 https://github.com/alibaba/ChatLearn.git
```

### Data

Please refer to [3-stage data](data.md) to prepare your training data.

## Step1: SFT

SFT refers to the process of fine-tuning a pre-trained language model using annotated dialogue data. In this example, we need to download the pre-trained model, and then start a simple SFT training demonstration.


### Download and Convert Pretrained Models

If using models from HuggingFace transformers, first download the pretraining checkpoint, such as the Bloom model from the HuggingFace Hub: `bigscience/bloom-7b1`, or pre-saved SFT models locally.
Then, use the following code to convert the HuggingFace transformers model into the Megatron-LM model format. In this example, we will convert the model to `TP (tensor_model_parallel_size)=8，PP (pipeline_model_parallel_size)=1` checkpoint, and the model will be stored in `MEGATRON_BLOOM_CKPT_PATH`.


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

### Start SFT Training

The script below is an example of SFT training. The `DATASET_PATH` is the path to the SFT training set, such as `$DATASET_ROOT/sft/train.jsonl`. In this example, we assume that the tokenizer's path is the same as the model checkpoint's path.

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm-chatlearn
cd ${CHATLEARN}/examples/megatron/step1_sft/

LOAD_PATH=$MEGATRON_BLOOM_CKPT_PATH \
TOKENIZER_PATH=$MEGATRON_BLOOM_CKPT_PATH \
DATASET_PATH=$DATASET_ROOT/sft/ \
bash bloom_sft.sh
```

The training logs and the completed models will be stored in `${CHATLEARN}/output/step1_sft` by default.
For specific definitions, please refer to the script `${CHATLEARN}/examples/megatron/step1_sft/bloom_sft.sh`.

For the environment variables and configurations required for distributed execution, please refer to [Distributed Execution](run.md).

## Step2: Reward Model Training

The Reward model refers to the model that serves as a proxy for human evaluation in RLHF. It provides real-time evaluation and scoring of the model's generated question responses. Given a question and model response, the Reward model produces a scalar representing the quality of the model's response.


### Start Reward Model Training

Based on InstructGPT[1], the Reward model training is initialized with the SFT model checkpoint. The training code is as follows:

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm-chatlearn
cd ${CHATLEARN}/examples/megatron/step2_reward/

LOAD_PATH=path-to-sft-ckpt \
TOKENIZER_PATH=$MEGATRON_BLOOM_CKPT_PATH \
DATASET_PATH=$DATASET_ROOT/rm/ \
bash bloom_reward.sh
```

The training logs and trained models will be saved by default in `${CHATLEARN}/output/step2_reward`. The specific definitions can be found in the script `${CHATLEARN}/examples/megatron/step2_reward/bloom_reward.sh`.

For the environment variables and configurations required for distributed execution, please refer to [Distributed Execution](run.md).

## Step 3: RLHF Training

RLHF refers to the process of trying different responses on a dataset consisting only of instructions and learning from the reward signals provided by a reward model for each response.

### Start RLHF Training

[Aliyun PAI DLC](https://www.aliyun.com/activity/bigdata/pai-dlc)[2] provides convenient and efficient support for RLHF training tasks. Below is a training script for a Bloom-7B Policy and a 7B Reward model. In this example, the user needs to set `POLICY_LOAD` to the checkpoint path produced by SFT (Supervised Fine-Tuning). The Policy model and Reference model will be initialized with the SFT checkpoint. `REWARD_LOAD` should be set to the checkpoint path produced by Reward training, and the user can specify the iteration number associated with the loaded checkpoint. The Reward model and Value model will be initialized using the Reward model weights. `VOCAB_FILE` should point to the folder containing the files required by `BloomTokenizer`.

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

7B Policy + 7B Reward RLHF training requires resources equivalent to 8 A100-80GB/A800-80GB/H800-80GB GPU cards.

For the environment variables and configurations required for distributed execution, please refer to [Distributed Execution](run.md).


### Evaluation

Firstly, we can use ChatLearn's model conversion tool to convert the Megatron-LM formatted model to HuggingFace's transformers model format.

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

We evaluated the performance of Bloom on the HH dataset, both after SFT and RLHF, using the GPT-4 API provided by MT-Bench. The results show that RLHF improves the average performance of the model compared to SFT. There is a significant improvement in the domains of Extraction, Math, Reasoning, STEM, and Writing. The performance gains observed here are due to the use of a Reward model trained on the open-source HH dataset. Customizing the Reward model contributes to achieving better results.

| Model | Coding | Extraction | Humanities | Math | Reasoning | Roleplay | STEM | Writing | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| bloom_sft | 1.45 | 1.1 | 3.35 | 1.45 | 2.6 | 3.1 | 2.65 | 1.4 | 2.27 |
| bloom_rlhf | 1.4 | **1.4** | 3.05 | **1.5** | **2.65** | 3.05 | **3.05** | **1.6** | **2.35** |

## Reference

1. Training language models to follow instructions with human feedback，[https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

