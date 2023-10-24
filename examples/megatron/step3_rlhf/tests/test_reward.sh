[ -z "$MEGATRON" ] && export MEGATRON=path-to-megatron
[ -z "$CHATLEARN" ] && export CHATLEARN=path-to-chatlearn
[ -z "$VOCAB_FILE" ] && export VOCAB_FILE=path-to-tokenizer
[ -z "$LOAD" ] && export LOAD=path-to-ckpt
[ -z "REWARD_LOAD_ITERATION" ] && export REWARD_LOAD_ITERATION=1000
[ -z "$DATASET_PATH" ] && export DATASET_PATH=path-to-dataset-json

export model_size=13B
source run_scripts/llama2/base_env.sh

eval_data_path=$DATASET_PATH \
tokenizer_model=$VOCAB_FILE \
reward_tp=8 \
reward_device=8 \
reward_load=$LOAD \
reward_load_iteration=$REWARD_LOAD_ITERATION \
python tests/test_reward_forward.py -c configs/llama2/test_reward.yaml
