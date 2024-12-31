#!/bin/bash

# Pretrain a multimodal model.
source activate /gf3/home/fjs/anaconda3/envs/megatron/

export CUDA=/gf3/softwares/cuda-11.8
export PATH=$CUDA/bin:$PATH
export LD_LIBRARY_PATH=$CUDA/lib64:$LD_LIBRARY_PATH
# CPUTI
export LD_LIBRARY_PATH=$CUDA/extras/CUPTI/lib64:$LD_LIBRARY_PATH
# nccl
export NCCL=$CUDA
# export NCCL_IB_SL=1
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL_NAME="llava-mistral-7b-clip336-pretraining"
WORKSPACE=/dat/fjs/llama_mistral/hete_mistral_clip_freeze_vit-tp4pp5
# Check that the user has set an output path for model checkpoints.
if [[ -z $WORKSPACE ]]; then
    echo "Please set WORKSPACE for storing your model checkpoints."
    exit 1
fi

SOURCE=`pwd`
OUTPUT_BASE="${WORKSPACE}/output"
OUTPUT="${OUTPUT_BASE}/${MODEL_NAME}"

LOGS_DIR="${OUTPUT}/logs"
TENSORBOARD_DIR="${OUTPUT}/tensorboard"

LOAD_NAME="mistral_clip336_tp4_pp5_freeze_vit"
if [[ -z $LOAD_NAME ]]; then
    echo "Please set LOAD_NAME for input model name."
    exit 1
fi


TOKENIZER_MODEL=/dat/fjs/llama_mistral/hf-mistral/
if [[ -z $TOKENIZER_MODEL ]]; then
    echo "Please set TOKENIZER_MODEL for tokenizer model name."
    exit 1
fi

FINETUNE_DIR="${WORKSPACE}/${LOAD_NAME}"
CHECKPOINT_DIR="${WORKSPACE}/${LOAD_NAME}"

DATA_TRAIN="${SOURCE}/examples/multimodal/pretrain_dataset.yaml"

DEBUG=1
if [[ $DEBUG -eq 1 ]]; then
    BZ=32
    NW=2
    HD=0.0
    LI=1
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
else
    BZ=256
    NW=2
    HD=0.1
    LI=5
    EXTRA_ARGS=""
    NONDETERMINISTIC_ATTN=1
fi

OPTIONS=" \
    --apply-layernorm-1p \
    --attention-softmax-in-fp32 \
    --normalization RMSNorm \
    --group-query-attention \
    --num-query-groups 8 \
    --no-masked-softmax-fusion \
    --num-workers ${NW} \
    --exit-duration-in-mins 230 \
    --untie-embeddings-and-output-weights \
    --disable-bias-linear \
    --position-embedding-type rope \
    --rotary-percent 1.0 \
    --rotary-base 1000000 \
    --swiglu \
    --attention-dropout 0.0 \
    --hidden-dropout ${HD} \
    --tensor-model-parallel-size 4 \
    --pipeline-model-parallel-size 5 \
    --split-spec "26,5,6,11,10"
    --num-layers 32 \
    --hidden-size 4096 \
    --num-attention-heads 32 \
    --seq-length 576 \
    --decoder-seq-length 1024 \
    --max-position-embeddings 4096 \
    --ffn-hidden-size 14336 \
    --train-iters 10 \
    --micro-batch-size 1 \
    --global-batch-size ${BZ} \
    --lr-decay-iters 20000 \
    --lr-warmup-fraction .01 \
    --lr 0.00015 \
    --min-lr 1.0e-5 \
    --lr-decay-style cosine \
    --tokenizer-type HuggingFaceTokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --data-path ${DATA_TRAIN} \
    --prompt-path ${SOURCE}/examples/multimodal/manual_prompts.json \
    --split 100,0,0 \
    --clip-grad 0.0 \
    --weight-decay 1e-2 \
    --adam-beta1 0.9 \
    --adam-beta2 0.95 \
    --init-method-std 0.014 \
    --eod-mask-loss \
    --patch-dim 14 \
    --img-h 336 \
    --img-w 336 \
    --dataloader-type external \
    --tensorboard-dir ${TENSORBOARD_DIR} \
    --language-model-type=mistral_7b \
    --disable-vision-class-token \
    ${EXTRA_ARGS} \
    --distributed-timeout-minutes 60 \
    --allow-missing-vision-projection-checkpoint \
    --no-load-optim \
    --no-load-rng \
    --log-interval ${LI} \
    --eval-iters 10 \
    --eval-interval 1000 \
    --use-flash-attn \
    --transformer-impl transformer_engine \
    --use-te \
    --timing-log-level 2 \
    --timing-log-option all \
    --freeze-ViT \
"
# --use-flash-attn \
# --transformer-impl transformer_engine \
# --use-te \
# --pretrained-checkpoint ${CHECKPOINT_DIR} \
# --load ${FINETUNE_DIR} \
# --use-checkpoint-args \
# --save ${FINETUNE_DIR} \
# --dataloader-save ${FINETUNE_DIR}/dataloader \
# --freeze-LM \
# --use-distributed-optimizer \
# --save-interval 1000 \
# --bf16 \
# --log-params-norm \
# --log-num-zeros-in-grad \
# --freeze-LM \


export NVTE_APPLY_QK_LAYER_SCALING=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=${NONDETERMINISTIC_ATTN}

GPUS_PER_NODE=4

# Change for multinode config
gn=`hostname | awk -F "n" '{print int($2)}'`
# node 3,9,51,52 2080 + 2080ti + 3090 + 3090
case $gn
        in 9)
        rank=0
        ;;
        2)
        rank=1
        ;;
        3)
        rank=2
        ;;
        49)
        rank=3
        ;;
        *)
        rank=4
esac

MASTER_ADDR=`scontrol show hostname $SLURM_NODELIST| head -n 3 | tail -n 1`
# MASTER_ADDR=`scontrol show hostname $SLURM_NODELIST| head -n 1`
MASTER_PORT=2234
NNODES=5
NODE_RANK=${rank:-"0"}
# NODE_RANK=$SLURM_PROCID
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

echo $NODE_RANK

torchrun $DISTRIBUTED_ARGS examples/multimodal/train.py ${OPTIONS}