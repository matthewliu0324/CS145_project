# -*- coding: utf-8 -*-

import os
import json
import torch
import argparse
import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from utils import IND4EVAL
from accelerate import Accelerator
from metric import compute_metric

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--lora_path', help='The path to the LoRA file', default="saved_dir/checkpoint-100")
parser.add_argument('--model_path', default='microsoft/phi-1_5')
parser.add_argument('--pub_path', help='The path to the pub file', default='/content/drive/MyDrive/Ucla/LLama3/dataset/pid_to_info_all.json')
parser.add_argument('--eval_path', default='/content/drive/MyDrive/Ucla/LLama3/dataset/ind_valid_author.json')
parser.add_argument('--saved_dir', default='eval_result')
args = parser.parse_args()

checkpoint = args.lora_path.split('/')[-1]

# Initialize Accelerator
accelerator = Accelerator()
device = accelerator.device

batch_size = 1

# Load model and tokenizer

tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(args.model_path, load_in_8bit=False, trust_remote_code=True).half()
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
lora_model = PeftModel.from_pretrained(model, args.lora_path).half().to(device)
print('Done loading PEFT model')

# Token IDs for evaluation
YES_TOKEN_IDS = tokenizer.convert_tokens_to_ids("yes")
NO_TOKEN_IDS = tokenizer.convert_tokens_to_ids("no")

# Load datasets
with open(args.pub_path, "r", encoding="utf-8") as f:
    pub_data = json.load(f)
with open(args.eval_path, "r", encoding="utf-8") as f: 
    eval_data = json.load(f)

eval_dataset = IND4EVAL(
    (eval_data, pub_data),
    tokenizer,
    max_source_length=25000,
    max_target_length=128,
)
print('Done reading dataset')

# Data collator function
def collate_fn(batch):
    batch = {k: [item[k] for item in batch] for k in ('input_ids', 'author', 'pub')}
    batch_input = tokenizer(
        batch['input_ids'],
        padding='longest',
        truncation=False,
        return_tensors="pt",
        add_special_tokens=False,
    )
    return batch_input, batch['author'], batch['pub']

# Create data loader
dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=batch_size, collate_fn=collate_fn)
val_data = accelerator.prepare(dataloader)

result = []

# Evaluation loop
with torch.no_grad():
    for index, batch in enumerate(tqdm.tqdm(val_data)):
        batch_input, author, pub = batch
        batch_input = {k: v.to(device) for k, v in batch_input.items()}

        response = model.generate(**batch_input, max_length=batch_input['input_ids'].shape[-1] + 16, return_dict_in_generate=True, output_scores=True)
        
        yes_prob, no_prob = response.scores[0][:, YES_TOKEN_IDS], response.scores[0][:, NO_TOKEN_IDS]
        logit = yes_prob / (yes_prob + no_prob)
        node_result = [(author[i], pub[i], logit[i].item()) for i in range(len(author))]
        batch_result = accelerator.gather_for_metrics(node_result)
        if accelerator.is_main_process:
            result.extend(batch_result)

# Save results
if accelerator.is_main_process: 
    if not os.path.exists(args.saved_dir):
        os.makedirs(args.saved_dir)
    res_list = {}
    for i in result:
        [aid, pid, logit] = i
        if aid not in res_list.keys():
            res_list[aid] = {}
        res_list[aid][pid] = logit
    with open(f'{args.saved_dir}/result-{checkpoint}.json', 'w') as f:
        json.dump(res_list, f)

print("Evaluation completed and results saved.")
