#!/bin/bash

[ -z "$MASTER_ADDR" ] && export MASTER_ADDR=localhost
[ -z "$WORLD_SIZE" ] && export WORLD_SIZE=1
[ -z "$GPUS_PER_NODE" ] && export GPUS_PER_NODE=4
[ -z "$RANK" ] && export RANK=0
[ -z "$MASTER_PORT" ] && export MASTER_PORT=12456

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE \
                  --nnodes $WORLD_SIZE \
                  --node_rank ${RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

echo $DISTRIBUTED_ARGS

# check the path
[[ -z "${MEGATRON}" ]] && { echo "MEGATRON path is not set"; exit 1; }
[[ -z "${CHATLEARN}" ]] && { echo "CHATLEARN path is not set"; exit 1; }
[[ -z "${LOAD_PATH}" ]] && { echo "LOAD_PATH is not set"; exit 1; }
[[ -z "${TOKENIZER_PATH}" ]] && { echo "TOKENIZER_PATH is not set"; exit 1; }
[[ -z "${DATASET_PATH}" ]] && { echo "DATASET_PATH is not set"; exit 1; }


export PYTHONPATH=${PYTHONPATH}:${MEGATRON}:${CHATLEARN}/examples/megatron


[ -z "$MODEL_SIZE" ] && export MODEL_SIZE=1B1

if [ $MODEL_SIZE = 1B1 ]; then

  NUM_LAYERS=24
  HIDDEN_SIZE=1536
  NUM_ATTN_HEADS=16
  INTERMEDIATE_SIZE=6144
  tp=4
  pp=1

elif [ $MODEL_SIZE = 1B7 ]; then

  NUM_LAYERS=24
  HIDDEN_SIZE=2048
  NUM_ATTN_HEADS=16
  INTERMEDIATE_SIZE=8192
  tp=4
  pp=1

elif [ $MODEL_SIZE = 7B1 ]; then

  NUM_LAYERS=30
  HIDDEN_SIZE=4096
  NUM_ATTN_HEADS=32
  INTERMEDIATE_SIZE=16384
  tp=8
  pp=1

fi

mb=8
gbs=$((mb * 8))
seq_len=2048

DIR=$(pwd)
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
mkdir -p $DIR/logs

NODE_RANK=$RANK
NNODES=$WORLD_SIZE



dp=$(($WORLD_SIZE * $GPUS_PER_NODE / $tp / $pp))
gbs=$(($gbs * $dp))


[ -z "$CHECKPOINT_PATH" ] && CHECKPOINT_PATH=${CHATLEARN}/output/step1_sft/bloom_hh_sft_$(date +%F)_gpt_${MODEL_SIZE}_${NNODES}w${GPUS_PER_NODE}g_tp${tp}_pp${pp}_mb${mb}_seqlen${seq_len}

mkdir -p $CHECKPOINT_PATH


MODEL_ARGS="--no-position-embedding --untie-embeddings-and-output-weights --use-alibi-position-embeddings --tokenizer-type AutoTokenizer --embed-layernorm"

log_file=$CHECKPOINT_PATH/stderr_$NODE_RANK.log

export CUDA_DEVICE_MAX_CONNECTIONS=1

python -m torch.distributed.launch $DISTRIBUTED_ARGS \
  finetune_sft.py \
  --tensor-model-parallel-size $tp \
  --pipeline-model-parallel-size $pp \
  --num-layers ${NUM_LAYERS} \
  --hidden-size ${HIDDEN_SIZE} \
  --num-attention-heads ${NUM_ATTN_HEADS} \
  --seq-length $seq_len \
  --max-position-embeddings 2048 \
  --micro-batch-size $mb \
  --global-batch-size $gbs \
  --train-iters 1000 --exit-interval 100000 \
  --lr-decay-iters 1000 \
  --lr-warmup-iters 40 \
  --lr 2.0e-5 \
  --min-lr 6.0e-12 \
  --lr-decay-style cosine \
  --log-interval 1 \
  --eval-iters 10 \
  --eval-interval 1000 \
  --data-path $DATASET_PATH/train.jsonl $DATASET_PATH/train.jsonl $DATASET_PATH/train.jsonl \
  --save-interval 100000 \
  --save $CHECKPOINT_PATH \
  --load $LOAD_PATH \
  --tensorboard-log-interval 100 \
  --split 98,2,0 \
  --clip-grad 1.0 \
  --weight-decay 0. \
  --adam-beta1 0.9 \
  --adam-beta2 0.999 \
  --init-method-std 0.006 \
  --tensorboard-dir $CHECKPOINT_PATH \
  --num-workers 4 \
  --vocab-file $TOKENIZER_PATH \
  --make-vocab-size-divisible-by 128 \
  --ffn-hidden-size $INTERMEDIATE_SIZE \
  --no-load-args \
  --no-load-rng \
  --no-load-optim \
  --log-timers-to-tensorboard \
  --log-batch-size-to-tensorboard \
  --log-validation-ppl-to-tensorboard \
  --dataloader-type cyclic \
  --bf16 \
  --use-distributed-optimizer \
  --adaptive-parallel-strategy-on-checkpoint \
  --sequence-parallel  \
  $MODEL_ARGS 2>&1 | tee -a ${log_file}
