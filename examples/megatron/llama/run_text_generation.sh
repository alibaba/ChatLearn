#!/bin/bash
set -e
ENV=$1
export CUDA_VISIBLE_DEVICES=0
MASTER_ADDR=localhost
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
GPUS_PER_NODE=1
NNODES=1
NODE_RANK=0
export CUDA_DEVICE_MAX_CONNECTIONS=1

MEGATRON_PATH=/mnt/user/E-xianyan.xianyanjia-189885/QWen2/
export PYTHONPATH=${MEGATRON_PATH}:$PYTHONPATH

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

CHECKPOINT_PATH=/mnt/user/E-xianyan.xianyanjia-189885/llama/megatron_models/llama-7b-ziya2/

MODEL_SIZE=13B
TP=1
BS=2
SEQ_LEN=1024
PAD_LEN=80
EXTRA_VOCAB_SIZE=16
EXTRA_VOCAB_SIZE=0
PR=bf16
TOP_K=0
INPUT_SEQ_LEN=512
OUTPUT_SEQ_LEN=512
INPUT_FILE=test.json
OUTPUT_FILE=test_out.json
TOP_P=0.85
TEMPERATURE=1
tokenizer_type=LLAMATokenizer #LLamaTokenizer #LLamaTokenizer-ziya

if [ $MODEL_SIZE = 7B ]; then

  NUM_LAYERS=32
  HIDDEN_SIZE=4096
  NUM_ATTN_HEADS=32
  INTERMEDIATE_SIZE=11008

elif [ $MODEL_SIZE = 13B ]; then

  NUM_LAYERS=40
  HIDDEN_SIZE=5120
  NUM_ATTN_HEADS=40
  INTERMEDIATE_SIZE=13824

fi

if [ $CHECKPOINT_PATH != none ]; then
  load_options=" \
		    --load $CHECKPOINT_PATH"
fi

if [ $INPUT_FILE = none ]; then
  input_options=" \
		               "
else
  input_options=" \
        --text-generate-output-file ${OUTPUT_FILE}\
        --text-generate-input-file ${INPUT_FILE} \
        "
fi

options="  \
        --micro-batch-size ${BS} \
        --num-layers ${NUM_LAYERS}  \
        --hidden-size ${HIDDEN_SIZE}  \
        --num-attention-heads ${NUM_ATTN_HEADS}  \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${SEQ_LEN} \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size 1 \
        --no-load-optim \
        --no-load-rng \
        --DDP-impl local \
        --top-p ${TOP_P} \
        --temperature ${TEMPERATURE}  \
        --top-k ${TOP_K} \
        --input-len ${INPUT_SEQ_LEN} \
        --out-seq-length ${OUTPUT_SEQ_LEN}  \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --tokenizer-type ${tokenizer_type} \
        --llama \
        --bf16
    "

#--max-padding-length ${PAD_LEN} \
#--swiglu \

run_cmd="python -m torch.distributed.launch $DISTRIBUTED_ARGS llama/generate_text_megatron_llama.py
 ${options} ${load_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
