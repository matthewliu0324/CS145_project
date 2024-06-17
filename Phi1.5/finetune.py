import logging
import os
import sys
import json
import torch
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    set_seed,
    Trainer,
    DataCollatorForSeq2Seq,
)
from peft import get_peft_model, LoraConfig, TaskType
from utils import INDDataSet
from arguments import ModelArguments, DataTrainingArguments, GLMTrainingArguments
from accelerate.utils import DistributedType

logger = logging.getLogger(__name__)

def main():
    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, GLMTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED
        
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)

    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}, "
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")
    set_seed(training_args.seed)

    config = AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    config.use_cache = False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    tokenizer = AutoTokenizer.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, config=config, trust_remote_code=True).to(device)

    if model_args.quantization_bit is not None:
        model = model.quantize(model_args.quantization_bit)

    # Load datasets
    with open(data_args.pub_data, "r", encoding="utf-8") as f:
        pub_data = json.load(f)
    with open(data_args.train_data, "r", encoding="utf-8") as f:
        train_data = json.load(f)

    train_dataset = INDDataSet(
        (train_data, pub_data),
        tokenizer,
        data_args.max_source_length,
        data_args.max_target_length,
    )

    # peft_config = LoraConfig(
    #     task_type=TaskType.CAUSAL_LM,
    #     inference_mode=False,
    #     r=model_args.lora_rank,
    #     target_modules=['query_key_value'],
    #     lora_alpha=model_args.lora_alpha,
    #     lora_dropout=model_args.lora_dropout,
    # )

    peft_config = LoraConfig(
        r=16,  # dimension of the updated matrices
        lora_alpha=64,  # parameter for scaling
        target_modules=[
          "q_proj",
          "up_proj",
          "o_proj",
          "k_proj",
          "down_proj",
          "gate_proj",
          "v_proj"],
        lora_dropout=0.1,  # dropout probability for layers
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config).to("cuda")

    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=-100,
        pad_to_multiple_of=None,
        padding=False,
    )

    training_args.push_to_hub = True

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )

    # Fine-tuning
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    trainer.train(resume_from_checkpoint=checkpoint)

    # Save model to Hugging Face Hub
    trainer.save_model()
    trainer.save_state()
    trainer.push_to_hub()

if __name__ == "__main__":
    main()
