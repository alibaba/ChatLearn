#!/bin/bash
# Convert LLaMA model from megatron format to huggingface format.
set -x

# config
chatlearn=${CHATLEARN}
megatron=${MEGATRON}
load_path=${LOAD_PATH}
save_path=${SAVE_PATH}
vocab_path=${VOCAB_PATH}
target_params_dtype=${target_params_dtype:-bf16}
temp_path=${save_path}/temp

set +x

# convert parallel strategy
START_TIME=$SECONDS

cd ${megatron}
python tools/checkpoint/convert.py \
    --model-type GPT \
    --loader megatron \
    --saver megatron \
    --target-tensor-parallel-size 1 \
    --target-pipeline-parallel-size 1 \
    --load-dir ${load_path} \
    --save-dir ${temp_path} \
    --megatron-path ${megatron}

# convert to hf format
cd ${chatlearn}
python chatlearn/tools/megatron_to_hf.py \
    --load_path ${temp_path} \
    --save_path ${save_path} \
    --target_params_dtype ${target_params_dtype} \
    --vocab_dir ${vocab_path} \
    --megatron_path ${megatron}

# clear temp path
rm -r $temp_path
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Conversion is done, using time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
