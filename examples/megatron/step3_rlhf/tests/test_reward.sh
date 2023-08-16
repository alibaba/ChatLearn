source run_scripts/base_env.sh

eval_data_path=$DATASET_PATH \
vocab_file=$VOCAB_FILE \
base_inference_tp=8 \
reward_device=8 \
reward_load=$REWARD_LOAD \
reward_load_iteration=$REWARD_LOAD_ITERATION \
python tests/test_reward_forward.py -c configs/llama/configs_13b_16g_pipe/test_reward.yaml
