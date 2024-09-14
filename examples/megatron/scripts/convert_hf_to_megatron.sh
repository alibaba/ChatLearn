#!/bin/bash
# Convert LLaMA2 model from huggingface format to megatron format.

set -x

# model config
# can be `gpt_llama' for GPT or Llama, or `mixtral' for Mixtral
model=${MODEL:-'gpt_llama'} 

# parallel config
tp=${TP:-1}
pp=${PP:-1}
ep=${EP:-1}

# checkpoint & tokenizer config
megatron=${MEGATRON}
chatlearn=${CHATLEARN}
export PYTHONPATH=${megatron}:${chatlearn}
load_dir=${LOAD_PATH}
save_dir=${SAVE_PATH}
tokenizer_model=${TOKENIZER_MODEL}
model_size=${model_size:-llama2-7B}

export CUDA_DEVICE_MAX_CONNECTIONS=1

set +x

# convert
START_TIME=$SECONDS

if [[ ${model} == 'gpt_llama' ]]; then
    cd ${chatlearn}
    python chatlearn/tools/megatron_checkpoint_utils.py \
        --model-type GPT \
        --loader llama_mistral \
        --checkpoint-type hf \
        --model-size ${model_size} \
        --saver megatron \
        --target-tensor-parallel-size ${tp} \
        --target-pipeline-parallel-size ${pp} \
        --load-dir ${load_dir} \
        --save-dir ${save_dir} \
        --tokenizer-model ${tokenizer_model}
elif [[ ${model} == 'mixtral' ]]; then
    # Mixtral can only be converted to mcore models.
    # Require Megatron-LM commit id >= c7a1f82.
    cd ${megatron}
    python tools/checkpoint/convert.py \
        --model-type GPT \
        --loader loader_mixtral_hf \
        --saver mcore \
        --target-tensor-parallel-size ${tp} \
        --target-pipeline-parallel-size ${pp} \
        --target-expert-parallel-size ${ep} \
        --load-dir ${load_dir} \
        --save-dir ${save_dir} \
        --tokenizer-model ${tokenizer_model}
fi

ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Conversion is done, using time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
