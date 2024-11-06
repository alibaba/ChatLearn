#!/bin/bash
set -x

[ -z "$MASTER_ADDR" ] && export MASTER_ADDR=localhost
[ -z "$WORLD_SIZE" ] && export WORLD_SIZE=1
[ -z "$GPUS_PER_NODE" ] && export GPUS_PER_NODE=8
[ -z "$RANK" ] && export RANK=0
[ -z "$MASTER_PORT" ] && export MASTER_PORT=12456

DISTRIBUTED_ARGS="--nproc_per_node ${GPUS_PER_NODE} \
                  --nnodes ${WORLD_SIZE} \
                  --node_rank ${RANK} \
                  --master_addr ${MASTER_ADDR} \
                  --master_port ${MASTER_PORT}"

# check the path
[[ -z "${MEGATRON}" ]] && { echo "MEGATRON path is not set"; exit 1; }
[[ -z "${CHATLEARN}" ]] && { echo "CHATLEARN path is not set"; exit 1; }
[[ -z "${LOAD_PATH}" ]] && { echo "LOAD_PATH is not set"; exit 1; }
[[ -z "${TOKENIZER_MODEL}" ]] && { echo "TOKENIZER_MODEL is not set"; exit 1; }
[[ -z "${DATASET_PATH}" ]] && { echo "DATASET_PATH is not set"; exit 1; }


export PYTHONPATH=${PYTHONPATH}:${MEGATRON}:${CHATLEARN}/examples/megatron:${CHATLEARN}

[ -z "$model_size" ] && export model_size="mixtral-8x7B"

if [ $model_size == "mixtral-8x7B" ]; then
  NUM_LAYERS=32
  HIDDEN_SIZE=4096
  NUM_ATTN_HEADS=32
  FFN_HIDDEN_SIZE=14336
  MAX_POSITION_EMBEDDINGS=32768
  NUM_QUERY_GROUPS=8
  NUM_EXPERTS=8
  MOE_ROUTER_TOPK=2
  seq_length=2048
  tp=1
  pp=4
  ep=8
  mb=1
  gbs=64
fi

DIR=$(pwd)
DATETIME=$(date +'date_%y-%m-%d_time_%H-%M-%S')
mkdir -p $DIR/logs

NODE_RANK=$RANK
NNODES=$WORLD_SIZE


dp=$(($WORLD_SIZE * $GPUS_PER_NODE / $tp / $pp))
gbs=$(($gbs * $dp))


[ -z "$CHECKPOINT_PATH" ] && CHECKPOINT_PATH=${CHATLEARN}/output/sft/mixtral_hh_sft_$(date +%F)_gpt_${model_size}_${NNODES}w${GPUS_PER_NODE}g_tp${tp}_pp${pp}_ep${ep}_mb${mb}_seqlen${seq_length}

mkdir -p $CHECKPOINT_PATH

MODEL_ARGS="
--disable-bias-linear \
--seq-length $seq_length \
--max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
--num-layers ${NUM_LAYERS} \
--hidden-size ${HIDDEN_SIZE} \
--ffn-hidden-size ${FFN_HIDDEN_SIZE} \
--num-attention-heads ${NUM_ATTN_HEADS} \
--init-method-std 0.006 \
--attention-dropout 0.0 \
--hidden-dropout 0.0 \
--normalization RMSNorm \
--position-embedding-type rope \
--swiglu \
--untie-embeddings-and-output-weights \
--group-query-attention \
--num-query-groups ${NUM_QUERY_GROUPS} \
--no-masked-softmax-fusion \
--no-position-embedding \
--transformer-impl transformer_engine \
--attention-softmax-in-fp32 "

MOE_ARGS="
--num-experts ${NUM_EXPERTS} \
--moe-router-topk ${MOE_ROUTER_TOPK} \
--moe-router-load-balancing-type aux_loss \
--moe-aux-loss-coeff 1e-2 \
--moe-token-dispatcher-type alltoall \
--overlap-param-gather \
--overlap-grad-reduce "

DATA_ARGS="
--tokenizer-type Llama2Tokenizer \
--tokenizer-model ${TOKENIZER_MODEL} \
--data-path $DATASET_PATH/train.jsonl $DATASET_PATH/train.jsonl $DATASET_PATH/train.jsonl \
--split 98,2,0 \
--dataloader-type cyclic "

TRAINING_ARGS="
--micro-batch-size $mb \
--global-batch-size $gbs \
--lr 2.0e-5 \
--train-iters 1000 \
--lr-decay-iters 1000 \
--lr-decay-style cosine \
--min-lr 6.0e-12 \
--weight-decay 0. \
--lr-warmup-iters 40 \
--clip-grad 1.0 \
--bf16 \
--exit-on-missing-checkpoint \
--use-checkpoint-args \
--adam-beta1 0.9 \
--adam-beta2 0.999 \
--use-flash-attn \
--finetune "

MODEL_PARALLEL_ARGS="
--tensor-model-parallel-size $tp \
--pipeline-model-parallel-size $pp \
--expert-model-parallel-size $ep \
--use-distributed-optimizer \
--sequence-parallel \
--distributed-timeout-minutes 60 \
"

LOGGING_ARGS="
--log-interval 1 \
--eval-iters 10 \
--eval-interval 1000 \
--save-interval 1000 \
--save $CHECKPOINT_PATH \
--load $LOAD_PATH \
--tensorboard-dir $CHECKPOINT_PATH \
--tensorboard-log-interval 100 \
--num-workers 8 \
--no-load-rng \
--no-load-optim \
--log-timers-to-tensorboard \
--log-batch-size-to-tensorboard \
--log-validation-ppl-to-tensorboard \
"

log_file=$CHECKPOINT_PATH/stderr_$NODE_RANK.log

export CUDA_DEVICE_MAX_CONNECTIONS=1

cd ${CHATLEARN}/examples/megatron/

torchrun $DISTRIBUTED_ARGS \
  entry/train_sft.py \
  ${MODEL_ARGS[@]} \
  ${MOE_ARGS[@]} \
  ${DATA_ARGS[@]} \
  ${TRAINING_ARGS[@]} \
  ${MODEL_PARALLEL_ARGS[@]} \
  ${LOGGING_ARGS[@]} 2>&1 | tee -a ${log_file} ; exit ${PIPESTATUS[0]}
