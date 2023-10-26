# End-to-end Training Tutorial with LLaMA Model

This document provides instructions for end-to-end training using the ChatLearn, Megatron-LM framework, and the LLaMA/LLaMA2 model.
It includes three stages of training: SFT, Reward, and RLHF training.


**The following is a collection of general environment variables used in this tutorial script:**

| ENV | Explanation                                                                                                                                                   |
| --- |---------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `CHATLEARN` | The location where the ChatLearn code is cloned [https://github.com/alibaba/ChatLearn.git](https://github.com/alibaba/ChatLearn.git)                          |
| `MEGATRON` | The location where the Megatron-LM code is cloned [https://github.com/NVIDIA/Megatron-LM.git](https://github.com/NVIDIA/Megatron-LM.git) |
| `DATASET_ROOT` | The root directory for storing the SFT/Reward/RLHF training dataset collection.                                                                               |
| `TOKENIZER_MODEL` | The path of `tokenizer.model` used by the Tokenizer.                                                                                                          |


# Setup: Image / Code and Data Preparation

## Docker Image

It is recommended to refer to `https://github.com/alibaba/ChatLearn/tree/master/docker/ngc/Dockerfile.ngc23.09` for preparing the docker image.

## Code

In this example, we need to download the following related code.

```bash
# Download the Megatron-LM code
git clone https://github.com/NVIDIA/Megatron-LM.git
git checkout 954a65b04
# Download the ChatLearn code
git clone https://github.com/alibaba/ChatLearn.git
```

## Data

Please refer to [3-stage data](data.md) to prepare your training data.

# Step1: SFT

SFT refers to the process of fine-tuning a pre-trained language model using annotated dialogue data. 
In this example, we need to download the pre-trained model, and then start a simple SFT training demonstration.


## 1.1 Download and Convert Pretrained Models

If you are using a model from HuggingFace transformers, you will first need to download the pre-trained checkpoint, 
such as the LLaMA2 model available on HuggingFace Hub (`meta-llama/Llama-2-7b-hf`), or a locally saved SFT model.
Then, you can use the following code to convert the HuggingFace transformers model into the Megatron-LM model format:

1. For the 7B model, we will convert the model into a checkpoint with `TP (tensor_model_parallel_size)=4` and `PP (pipeline_model_parallel_size)=1`,
and the model will be saved in `MEGATRON_LLAMA_CKPT_PATH`.
2. For the 13B model, we will convert the model into a checkpoint with `TP=8` and `PP=1`.
3. For the 70B model, we will convert the model into a checkpoint with `TP=8` and `PP=4`.

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

## 1.2 Start SFT Training

The script below is an example of SFT training. The `DATASET_PATH` is the path to the SFT training set, such as `$DATASET_ROOT/sft/train.jsonl`. 
The `MODEL_SIZE` is an environment variable specified in the script to indicate the size of the model, which can be `7B`, `13B`, or `70B`.

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

The training logs and the completed models will be stored in `${CHATLEARN}/output/step1_sft` by default.
For specific definitions, please refer to the script `${CHATLEARN}/examples/megatron/step1_sft/llama_sft2.sh`.

In our training script, the resource requirements (assuming the resources are A100-80GB/A800-80GB/H800-80GB GPUs) are as follows:
1. 7B SFT: 8 GPUs
2. 13B SFT: 8 GPUs
3. 70B SFT: 4*8 GPUs

For the environment variables and configurations required for distributed execution, please refer to [Distributed Execution](run.md).

# Step2: Reward Model Training

The Reward model refers to the model that serves as a proxy for human evaluation in RLHF.
It provides real-time evaluation and scoring of the model's generated question responses.
Given a question and model response, the Reward model produces a scalar representing the quality of the model's response.


## 2.1 Start Reward Model Training

Based on InstructGPT[1], the Reward model training is initialized with the SFT model checkpoint. The training code is as follows:

```bash
export CHATLEARN=path-to-chatlearn
export MEGATRON=path-to-megatron-lm-extension
cd ${CHATLEARN}/examples/megatron/step2_reward/

LOAD_PATH=path-to-sft-ckpt \
TOKENIZER_MODEL=$LLAMA2_TOKENIZER_MODEL \
DATASET_PATH=$DATASET_ROOT/rm/ \
bash llama2_reward.sh
```

The training logs and trained models will be saved by default in `${CHATLEARN}/output/step2_reward`. 
The specific definitions can be found in the script `${CHATLEARN}/examples/megatron/step2_reward/llama_reward2.sh`.

The resource requirements for training a reward model of the same scale are the same as those for SFT models.

For the environment variables and configurations required for distributed execution, please refer to [Distributed Execution](run.md).

# Step 3: RLHF Training

RLHF refers to the process of trying different responses on a dataset consisting only of instructions and learning from the reward signals provided by a reward model for each response.

## 3.1 Start RLHF Training

Here is a training script for LLaMA2-7B Policy and 7B Reward models.
In this example, the user needs to set `POLICY_LOAD` to the checkpoint path generated by SFT.
The Policy and Reference models will be initialized with the SFT checkpoint.
`REWARD_LOAD` should be set to the checkpoint path generated by the Reward training, and the user can specify the iteration number for the loaded checkpoint.
The Reward and Value models will be initialized with the weights of the Reward model.
`TOKENIZER_MODEL` should be set to the folder path where the `tokenizer.model` for LlamaTokenizer is located.

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

If you need to train a 13B / 70B model, simply replace `run_7b_7b.sh` in the above training script with `run_13b_13b.sh` / `run_70b_70b.sh`.
You can also modify the model configuration and other parameters according to your needs.

In our training script, the resource requirements (assuming the resources are A100-80GB / A800-80GB / H800-80GB GPUs) are as follows:

1. 7B RLHF: 8 GPUs
2. 13B RLHF: 2*8 GPUs
3. 70B RLHF: 4*8 GPUs

For the environment variables and configurations required for distributed execution, please refer to [Distributed Execution](run.md).

Note that for RLHF tasks, if you are running on PAI DLC, you need to fill in the advanced configuration `customPortList=30000-30050,createSvcForAllWorkers=true`.

## 3.2 Evaluation

Firstly, we can use ChatLearn's model conversion tool to convert the Megatron-LM formatted model to HuggingFace's transformers model format.

```bash
cd $CHATLEARN
python chatlearn/tools/megatron_to_hf.py \
  --load_path ${dir-to-megatron-model} \
  --save_path ${save-dir} \
  --target_params_dtype bf16 \
  --vocab_dir ${dir-of-vocab-file} \
  --megatron_path ${dir-to-megatron}
```

We evaluated the performance of LLaMA-13B on the HH dataset, both after SFT and RLHF, using the GPT-4 API provided by MT-Bench. The results show that RLHF improves the average performance of the model compared to SFT. There is a significant improvement in the domains of Humanities, Math, Roleplay, STEM, and Writing. The performance gains observed here are due to the use of a Reward model trained on the open-source HH dataset. Customizing the Reward model contributes to achieving better results.

| Model | Coding | Extraction | Humanities | Math | Reasoning | Roleplay | STEM | Writing | Avg |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| llama_sft | 1.6 | 2.7 | 4.2 | 1.1 | 2.85 | 3.35 | 4.55 | 2.95 | 2.90 |
| llama_rlhf | **1.75** | **3.45** | **4.75** | **1.55** | **3.5** | **5.85** | **5.0** | **5.0** | **3.85** |

# Reference

1. Training language models to follow instructions with human feedback，[https://arxiv.org/abs/2203.02155](https://arxiv.org/abs/2203.02155)

