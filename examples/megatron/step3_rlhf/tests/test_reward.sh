export VOCAB_FILE=<vocab_file_path>
export REWARD_LOAD=<reward_model_checkpoint_path>
export REWARD_LOAD_ITERATION=1000
export DATASET_PATH=<eval_data_path>

export model_size=13B
source run_scripts/llama/base_env.sh

eval_data_path=$DATASET_PATH \
vocab_file=$VOCAB_FILE \
base_inference_tp=8 \
reward_device=8 \
reward_load=$REWARD_LOAD \
reward_load_iteration=$REWARD_LOAD_ITERATION \
python tests/test_reward_forward.py -c configs/llama/test_reward.yaml
