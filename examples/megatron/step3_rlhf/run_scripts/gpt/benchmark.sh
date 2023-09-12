
DATA_DIR=${CHATLEARN}/output/gpt/
mkdir -p $DATA_DIR/logs

label=$(date +%F)_rlhf_${model_size}_${max_new_tokens}_${num_device}g_lora-${lora}-policy-tp${policy_tp}_pp${ppo_policy_pp}_reward-tp${reward_tp}-pp${ppo_reward_pp}_bs${policy_generation_batch_size}_${ref_generation_bs}_${reward_generatiob_bs}_${value_generation_bs}_${train_micro_batch_size}_${train_global_batch_size}_device_${num_device_ref}_${num_device_reward}_${num_device_value}_ranking_${batch_generation_ranking}_min-${min_prompt_length}_${sample_per_episode}_thred${inference_batch_times_seqlen_threshold}_refpp${ref_pp}_rewardpp${reward_pp}_gc_${policy_recompute_granularity}
if [[ ! -f "${DATA_DIR}/rm_static_train.jsonl" ]]; then
    wget -P $DATA_DIR http://odps-release.cn-hangzhou.oss.aliyun-inc.com/torchacc/accbench/datasets/opensource/rlhf/rm_static_train.jsonl
fi
if [[ ! -f "${DATA_DIR}/gpt2-vocab.json" ]]; then
    wget -P $DATA_DIR https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-merges.txt
    wget -P $DATA_DIR https://s3.amazonaws.com/models.huggingface.co/bert/gpt2-vocab.json
fi
# use_eod_token_for_early_termination=False is for benchmark only, should set to True in real run
use_eod_token_for_early_termination=False \
enable_lora_value=${lora} \
enable_lora_policy=${lora} \
data_path=${DATA_DIR}/rm_static_train.jsonl \
vocab_file=${DATA_DIR}/gpt2-vocab.json \
merge_file=${DATA_DIR}/gpt2-merges.txt \
python train_rlhf.py -c configs/gpt/rlhf.yaml 2>&1 | tee ${DATA_DIR}/logs/${label}.log
