#!/bin/bash
# Convert LLaMA model from megatron format to huggingface format.
set -ex
set pipefail

# path config
chatlearn=${CHATLEARN}
megatron=${MEGATRON}
load_path=${LOAD_PATH}
save_path=${SAVE_PATH}
vocab_path=${VOCAB_PATH}
target_params_dtype=${target_params_dtype:-bf16}
temp_path=${save_path}/temp

# model config
# can be `gpt_llama' for GPT or Llama, or `mixtral' for Mixtral
model=${MODEL:-'gpt_llama'}

# Whether to use legacy models, default: True
use_legacy_models=${USE_LEGACY_MODELS:-"True"}
if [[ ${use_legacy_models} == "False" ]]; then
    if [[ ${model} == 'gpt_llama' ]]; then
        # TODO: migrate to mcore
        loader_ckpt_format="mcore"
        saver_ckpt_format="megatron"
    elif [[ ${model} == 'mixtral' ]]; then
        loader_ckpt_format="mcore_mixtral"
        saver_ckpt_format="mcore"
    else
        echo -e "\033[31m Unrecognized model ${model} \033[0m"
        exit -1
    fi
    MCORE_ARGS=""
else
    loader_ckpt_format="megatron"
    saver_ckpt_format="megatron"
    MCORE_ARGS="--use_legacy_models"
fi

set +x

# convert parallel strategy
START_TIME=$SECONDS

if [[ ! -d ${temp_path} ]]; then
    if [[ ${model} == 'gpt_llama' ]]; then
        cd ${megatron}
        python tools/checkpoint/convert.py \
            --model-type GPT \
            --loader ${loader_ckpt_format} \
            --saver ${saver_ckpt_format} \
            --target-tensor-parallel-size 1 \
            --target-pipeline-parallel-size 1 \
            --load-dir ${load_path} \
            --save-dir ${temp_path} \
            --megatron-path ${megatron}
    elif [[ ${model} == 'mixtral' ]]; then
        cd ${chatlearn}
        export PYTHONPATH=${chatlearn}:${megatron}:${megatron}/tools/checkpoint:${PYTHONPATH}
        python chatlearn/tools/convert.py \
            --model-type GPT \
            --loader ${loader_ckpt_format} \
            --saver-prefix tools.checkpoint.saver \
            --saver ${saver_ckpt_format} \
            --target-tensor-parallel-size 1 \
            --target-pipeline-parallel-size 1 \
            --target-expert-parallel-size 1 \
            --load-dir ${load_path} \
            --save-dir ${temp_path} \
            --megatron-path ${megatron}
    else
        echo -e "\033[31m Unrecognized model ${model} \033[0m"
        exit -1
    fi
fi

if [[ $? != 0 ]]; then
    exit $?
fi

# convert to hf format
cd ${chatlearn}
python chatlearn/tools/megatron_to_hf.py \
    --load_path ${temp_path} \
    --save_path ${save_path} \
    --target_params_dtype ${target_params_dtype} \
    --vocab_dir ${vocab_path} \
    --megatron_path ${megatron} \
    ${MCORE_ARGS}

# clear temp path
rm -r $temp_path
ELAPSED_TIME=$(($SECONDS - $START_TIME))
echo "Conversion is done, using time: $(($ELAPSED_TIME / 60)) min $(($ELAPSED_TIME % 60)) sec"
