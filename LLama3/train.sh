# Configuration Parameters
LR=3e-5
NUM_GPUS=1  # Set to 1 for debugging; increase if running on a Linux machine with multiple GPUs
LORA_RANK=32
LORA_ALPHA=64
LORA_DROPOUT=0.05
MAX_SOURCE_LEN=25000
MAX_TARGET_LEN=16
DEV_BATCH_SIZE=1
GRAD_ACCUMULATION_STEPS=16
EPOCH=4
SAVE_INTERVAL=250
WARMUP_RATIO=0.03
SCHEDULER=cosine
RUN_NAME=text
BASE_MODEL_PATH=meta-llama/Meta-Llama-3-8B
PUB_PATH=./dataset/pid_to_info_all.json
TRAIN_PATH=./dataset/train_author.json
DATESTR=$(date +%Y%m%d-%H%M%S | tr -d '\r')
OUTPUT_DIR=output/${RUN_NAME}-${DATESTR}-${LR}
MASTER_PORT=$(shuf -n 1 -i 10000-65535)
# Create output directory
mkdir -p $OUTPUT_DIR
# Run training script with python (single GPU setup for debugging)
python3 finetune.py --model_name_or_path "meta-llama/Meta-Llama-3-8B" --output_dir $OUTPUT_DIR --train_format input-output --pub_data $PUB_PATH --train_data $TRAIN_PATH --lora_rank $LORA_RANK --lora_alpha $LORA_ALPHA --lora_dropout $LORA_DROPOUT --max_source_length $MAX_SOURCE_LEN --max_target_length $MAX_TARGET_LEN --preprocessing_num_workers 1 --per_device_train_batch_size $DEV_BATCH_SIZE --gradient_accumulation_steps $GRAD_ACCUMULATION_STEPS --warmup_ratio $WARMUP_RATIO --num_train_epochs $EPOCH --logging_steps 1 --save_steps $SAVE_INTERVAL --learning_rate $LR --bf16 --push_to_hub --deepspeed configs/deepspeed.json 2>&1 | tee ${OUTPUT_DIR}/train.log