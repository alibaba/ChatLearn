# End-to-end Training Tutorial with Llama Model

This document provides instructions for end-to-end training using the ChatLearn, Megatron-LM and vLLM framework, and the Llama/Llama2 model. ChatLearn supports three training policies as follows:
1. RLHF(Reinforcement Learning from Human Feedback): which includes three stages of training: SFT, Reward, and RLHF training.
2. Direct Preference Optimization(DPO): which includes two stages of training: SFT and DPO training.
3. OnlineDPO/GRPO: which fall in between RLHF and DPO, includes three stages of training: SFT, Reward, and DPO training.

**The following is a collection of general environment variables used in this tutorial script:**

| ENV | Explanation                                                                                                                                                   |
| --- |---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CHATLEARN` | The location where the ChatLearn code is cloned [https://github.com/alibaba/ChatLearn.git](https://github.com/alibaba/ChatLearn.git)                          |
| `MEGATRON` | The location where the Megatron-LM code is cloned [https://github.com/NVIDIA/Megatron-LM.git](https://github.com/NVIDIA/Megatron-LM.git) |
| `DATASET_ROOT` | The root directory for storing the SFT/Reward/RLHF/DPO/OnlineDPO/GRPO training dataset collection.                                                                               |
| `TOKENIZER_MODEL` | The path of `tokenizer.model` used by the Tokenizer.                                                                                                          |


## Setup: Image / Code and Data Preparation

### Image / Code

Please refer to [Environment and Code Setup](../installation.md).

### Data

Please refer to [3-stage data](data.md) to prepare your training data.

## SFT

SFT refers to the process of fine-tuning a pre-trained language model using annotated dialogue data. 
In this example, we need to download the pre-trained model, and then start a simple SFT training demonstration.


### Download and Convert Pretrained Models

If you are using a model from HuggingFace transformers, you will first need to download the pre-trained checkpoint, 
such as the Llama2 model available on HuggingFace Hub (`meta-llama/Llama-2-7b-hf`), or a locally saved SFT model.
Then, you can use the following code to convert the HuggingFace transformers model into the Megatron-LM model format:

1. For the llama2-7B model, we will convert the model into a checkpoint with `TP (tensor_model_parallel_size)=4` and `PP (pipeline_model_parallel_size)=1`,
and the model will be saved in `SAVE_PATH`.
2. For the llama2-13B model, we will convert the model into a checkpoint with `TP=8` and `PP=1`.
3. For the llama2-70B model, we will convert the model into a checkpoint with `TP=8` and `PP=4`.

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

### Start SFT Training

The script below is an example of SFT training. The `DATASET_PATH` is the path to the SFT training set, such as `$DATASET_ROOT/sft/train.jsonl`. 
The `model_size` is an environment variable specified in the script to indicate the size of the model, which can be `llama2-7B`, `llama2-13B`, or `llama2-70B`.

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

The training logs and the completed models will be stored in `${CHATLEARN}/output/sft` by default.
For specific definitions, please refer to the script `${CHATLEARN}/examples/megatron/scripts/train_sft_llama.sh`.

In our training script, the resource requirements (assuming the resources are A100-80GB/A800-80GB GPUs) are as follows:
1. llama2-7B SFT: 8 GPUs
2. llama2-13B SFT: 8 GPUs
3. llama2-70B SFT: 4*8 GPUs

For the environment variables and configurations required for distributed execution, please refer to [Distributed Execution](run.md).

## Reward Model Training

The Reward model refers to the model that serves as a proxy for human evaluation in RLHF.
It provides real-time evaluation and scoring of the model's generated question responses.
Given a question and model response, the Reward model produces a scalar representing the quality of the model's response.

**Hint**: No need of reward model training for DPO mode.

### Start Reward Model Training

Based on InstructGPT[1], the Reward model training is initialized with the SFT model checkpoint. The training code is as follows:

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm
cd ${CHATLEARN}/examples/megatron/

LOAD_PATH=path-to-sft-ckpt \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
DATASET_PATH=$DATASET_ROOT/rm/ \
bash scripts/train_reward_llama.sh
```

The training logs and trained models will be saved by default in `${CHATLEARN}/output/reward`. 
The specific definitions can be found in the script `${CHATLEARN}/examples/megatron/scripts/train_reward_llama.sh`.

The resource requirements for training a reward model of the same scale are the same as those for SFT models.

For the environment variables and configurations required for distributed execution, please refer to [Distributed Execution](run.md).

## Alignment Training

ChatLearn supports multiple alignments: RLHF, DPO, OnlineDPO, GRPO
 
### Start Alignment Training

Take Llama2-7B for example as follows.

#### RLHF

Here is a training script for Llama2-7B Policy and 7B Reward models.
In this example, the user needs to set `POLICY_LOAD` to the checkpoint path generated by SFT, which used for Policy and Value model.
The Policy and Reference models will be initialized with the SFT checkpoint.
`REWARD_LOAD` should be set to the checkpoint path generated by the Reward training, and the user can specify the iteration number for the loaded checkpoint.
The Reward and Value models will be initialized with the weights of the Reward model.
`TOKENIZER_MODEL` should be set to the folder path where the `tokenizer.model` for Llama2Tokenizer is located.

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
bash run_scripts/train_rlhf_llama.sh
```

#### OnlineDPO/GRPO

OnlineDPO/GRPO training process is similar to RLHF.

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

Here is a training script for Llama2-7B Policy and 7B Reward models.
In this example, the user needs to set `POLICY_LOAD` to the checkpoint path generated by SFT.
The Policy and Reference models will be initialized with the SFT checkpoint.
`TOKENIZER_MODEL` should be set to the folder path where the `tokenizer.model` for Llama2Tokenizer is located.

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

To train a GRPO Math model, first refer to [Math data](data.md#Math) to prepare the mathematics dataset. Below is an example of training a Llama2-7B model.

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

#### Models of Other Sizes

If you need to train a llama2-13B / llama2-70B model, simply change `export model_size=llama2-7B` with `export model_size=llama2-13B` / `export model_size=llama2-70B`.
You can also modify the model configuration and other parameters according to your needs.

In our training script, the resource requirements (assuming the resources are A100-80GB / A800-80GB GPUs) are as follows:

1. llama2-7B RLHF: 8 GPUs
2. llama2-13B RLHF: 2*8 GPUs
3. llama2-70B RLHF: 4*8 GPUs

For the environment variables and configurations required for distributed execution, please refer to [Distributed Execution](run.md).


### Evaluation

Firstly, we can use ChatLearn's model conversion tool to convert the Megatron-LM (Legacy) formatted model to HuggingFace's transformers model format.

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

We evaluated the performance of Llama2-7B on the HH dataset, after training with SFT, RLHF, DPO or training with OnlineDPO, using the GPT-4 API provided by MT-Bench. The results show that alignment training (RLHF, DPO, and OnlineDPO) by ChatLearn improves the average performance of the model compared to SFT. For RLHF, there is a significant improvement in the domains of Humanities, Math, Roleplay, Reasoning, and Writing. The performance gains observed here are due to the use of a Reward model trained on the open-source HH dataset. Customizing the Reward model contributes to achieving better results.

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

### Using Megatron-Core Model Format

If you need to train a model in the Megatron-Core format, you only need to add the environment variable `USE_LEGACY_MODELS=False` to each of the above scripts. This variable controls whether Megatron-LM selects the Megatron-Core model format for training. Its default value is `True`, indicating that Megatron-LM defaults to the Legacy model format. Setting it to `False` indicates that Megatron-LM selects the Megatron-Core model format.

#### Usage Example

Using the example of training an Llama-7B model with the RLHF training mode, the steps for SFT, Reward, and RLHF training are as follows:

1. Convert a HuggingFace transformers model to the Megatron-Core model format

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

2. Start SFT training

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

3. Start Reward training

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

4. Start RLHF training

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

#### Evaluation

Firstly, we can use ChatLearn's model conversion tool to convert the Megatron-Core formatted model to HuggingFace's transformers model format.

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

We evaluated the performance of Llama2-7B on the HH dataset, after training with SFT and RLHF, using the GPT-4 API provided by MT-Bench. The results show that RLHF by ChatLearn improves the average performance of the model compared to SFT. For RLHF, there is a significant improvement in the domains of Humanities, Roleplay, STEM, and Writing. The performance gains observed here are due to the use of a Reward model trained on the open-source HH dataset. Customizing the Reward model contributes to achieving better results.


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

1. Training language models to follow instructions with human feedback [https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

