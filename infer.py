import os
from transformers import *
import torch
from tqdm import tqdm, tqdm_notebook
import numpy as np
import pandas as pd
import math
import json
import time
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('--model_name_or_path', type=str)
parser.add_argument('--checkpoint_path', type=str)
parser.add_argument('--owner', type=str)
args = parser.parse_args()

tokenizer = RobertaTokenizer.from_pretrained(args.model_name_or_path)
tokenizer.add_tokens(["</s>", "<s>"], special_tokens=True)
if not tokenizer._pad_token:
    print("Adding pad token to tokenizer")
    tokenizer.add_special_tokens({"pad_token": "<pad>"})
model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path)
model.resize_token_embeddings(len(tokenizer))
_ = model.eval()
_ = model.cuda()
model.load_state_dict(torch.load(args.checkpoint_path,map_location='cpu'), strict=False)

@torch.no_grad()
def inference(input_ids,repetition_penalty=2.5,temperature=1.0,deterministic=False,n_sequences=3,do_postprocess=False):
    if deterministic:
        model.eval()
    else:
        model.train()
    #print(temperature)
    x = model.generate(input_ids,temperature=temperature, max_length=128, do_sample=False, early_stopping=True, 
                       eos_token_id=tokenizer.pad_token_id, pad_token_id=tokenizer.pad_token_id,
                       no_repeat_ngram_size=3, top_k=10, repetition_penalty=repetition_penalty, num_return_sequences=n_sequences, num_beams=max(n_sequences,3))
    outputs = []
    for seq in x:
        outputs.append(tokenizer.decode([w for w in seq if w != tokenizer.pad_token_id]))
    return outputs

print("\n===========================================\n")
print("\n====================CHAT===================\n")
print("\n===========================================\n")

conv = []
while True:
    x = input("You: ")
    conv.append({"owner": False, "content": x})
    context = conv[-10:]
    line = ""
    for x in context:
        if x['owner']:
            line += f"</s> {x['content']}"
        else:
            line += f"<s> {x['content']}"
    line += "</s>"
    input_ids = tokenizer.encode(line, add_special_tokens=False)[-80:]
    input_ids = torch.tensor([input_ids]).cuda()
    outputs = np.array([x for x in inference(input_ids, deterministic=False, n_sequences=5) if len(x) > 0])
    
    n_msgs = len([x for x in line.replace("</s>","<s>").split("<s>") if len(x) > 0])
    response = np.random.choice(outputs)
    response = [x for x in response.replace("</s>","<s>").split("<s>") if len(x) > 0][n_msgs].strip()
    
    print(f"Bot {args.owner}:", response, "\n")
    conv.append({"owner": True, "content": response})
