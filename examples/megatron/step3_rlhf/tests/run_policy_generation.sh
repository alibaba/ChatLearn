

TP=8
scripts=gpt

if [[ "$scripts" == "gpt" ]]; then
    configs=configs/gpt/rlhf_inference.yaml
elif [[ "$scripts" == "llama" ]]; then
    configs=configs/llama/test_policy.yaml
else
    echo "unexpected scripts $scripts."
    exit 1
fi

source run_scripts/$scripts/base_env.sh
export exp_name=run_test_13b_tp${TP}_meg_$scripts
mkdir -p logs

vocab_file=$VOCAB_FILE \
merge_file=$MERGE_FILE \
generation_batch_size=64 \
num_device=$TP \
tp=$TP \
eval_data_path=$DATASET_PATH \
policy_inference_load=$LOAD \
eval_output_dir=${Megatron}/logs/$(date +%F) \
python tests/test_policy_generation.py -c $configs 2>&1 | tee logs/${exp_name}.log
