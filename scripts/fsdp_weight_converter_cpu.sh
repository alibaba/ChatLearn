export CHATLEARN=$(pwd)
python chatlearn/offline_ckpt_converter.py \
    --hf_dir ${CHATLEARN}/Qwen3-235B-A22B/ \
    --ckpt_dir ${CHATLEARN}/output/qwen3-grpo-235b-a22b/save_model/policy_trainer \
    --save_dir ${CHATLEARN}/output/qwen3-grpo-8b/save_model/huggingface/ \
    --iter 200 \
    --groupgemm