#!/bin/bash
set -exo
set pipefail

export CHATLEARN=${CHATLEARN:-"path-to-chatlearn"}
export MEGATRON=${MEGATRON:-"path-to-megatron-lm"}
export LOAD_PATH=${LOAD_PATH:-"path-to-hf-ckpt"}
export TEMP_PATH=${TEMP_PATH:-"path-to-converted-mg-ckpt"}
export SAVE_PATH=${SAVE_PATH:-"path-to-converted-back-hf-ckpt"}
export VOCAB_PATH=${VOCAB_PATH:-"path-to-vocabulary"}
export TOKENIZER_MODEL=${TOKENIZER_MODEL:-"path-to-tokenizer-model"}

export MODEL=${MODEL:-"mixtral"}
export USE_LEGACY_MODELS=${USE_LEGACY_MODELS:-"False"}

# Step 1: Convert to Megatron checkpoint

cd $CHATLEARN/examples/megatron/

TP=1 \
PP=4 \
EP=8 \
LOAD_PATH=${LOAD_PATH} \
SAVE_PATH=${TEMP_PATH} \
bash scripts/convert_hf_to_megatron.sh

# Step 2: Convert to HuggingFace checkpoint

LOAD_PATH=${TEMP_PATH} \
SAVE_PATH=${SAVE_PATH} \
VOCAB_PATH=${VOCAB_PATH} \
target_params_dtype=bf16 \
bash scripts/convert_megatron_to_hf.sh

# Step 3: Compare converted hf ckpt against the original hf ckpt

python3 tests/test_checkpoint_conversion.py \
    --src-path ${LOAD_PATH} \
    --dst-path ${SAVE_PATH}

if [[ $? != 0 ]]; then
    echo -e "\033[31m Unrecognized model ${model} \033[0m"
    exit -1
fi

rm -rf ${TEMP_PATH}
rm -rf ${SAVE_PATH}

echo "Test success!"
