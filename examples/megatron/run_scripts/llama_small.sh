source base_env.sh

exp_base_dir=/cpfs01/user/E-xianyan.xianyanjia-189885/experiments/

export exp_name=$(date +%F)_small_test
export tensorboard_dir=${exp_base_dir}/tensorboard/${exp_name}
exp_dir=${exp_base_dir}/${exp_name}

export save_dir=${exp_dir}/saved_models
export log_dir=${exp_dir}/logs/
export data_checkpoint_path=${exp_dir}/data_checkpoint/
export save_episode_interval=100
export eval_episode_interval=100
export eval_output_dir=${exp_dir}/eval_output_dir

mkdir $log_dir
# export CUDA_VISIBLE_DEVICES=0
export continue_train=0
export continue_train_global_batch_size=8
export continue_inference_instances=1
export continue_inference_batch_size=4

build_path=build \
  eval_data_num_limit=8 \
  do_math_eval=1 \
  ngram_coef=0 \
  lm_coef=0 \
  math_coef=0 \
  raw_reward_coeff=1 \
  data_path=/mnt/shared/Group-m6/tianhang_zhu/chatgpt_api_v2/chatgpt_api_v2/all_rlhf_data/train_plus_test_use.jsonl \
  eval_data_path=/mnt/shared/Group-m6/tianhang_zhu/chatgpt_api_v2/chatgpt_api_v2/all_rlhf_data/test_use.jsonl \
  python train_rlhf.py -c llama/configs_small/rlhf.yaml 2>&1 | tee -a ${log_dir}/log.txt
