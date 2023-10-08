export MEGATRON=path-to-megatron
export CHATLEARN=path-to-chatlearn

TP=4
model_type=llama # gpt/llama
VOCAB_FILE=path-to-tokenizer-dir
LOAD=path-to-ckpt
DATASET_PATH=path-to-dataset-json
OUTPUT=$CHATLEARN/output/tests/

export model_size=13B # model size

if [[ "$model_type" == "gpt" ]]; then
    configs=configs/gpt/test_policy.yaml
    export vocab_file=$VOCAB_FILE
    export merge_file=$MERGE_FILE
elif [[ "$model_type" == "llama" ]]; then
    configs=configs/llama/test_policy.yaml
    export vocab_file=$VOCAB_FILE
else
    echo "unexpected model_type $model_type."
    exit 1
fi

source run_scripts/$model_type/base_env.sh

export exp_name=run_test_${model_size}_tp${TP}_meg_$model_type

generation_batch_size=64 \
num_device=$TP \
policy_tp=$TP \
eval_data_path=$DATASET_PATH \
policy_inference_load=$LOAD \
eval_output_dir=$OUTPUT \
python tests/test_policy_generation.py -c $configs 2>&1 | tee ${OUTPUT}/${exp_name}.log
