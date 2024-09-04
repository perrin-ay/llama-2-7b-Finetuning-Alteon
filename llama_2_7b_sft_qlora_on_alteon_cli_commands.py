# -*- coding: utf-8 -*-
"""Llama-2-7b SFT QLORA on Alteon cli commands

"""

import bitsandbytes
from accelerate import init_empty_weights
from transformers import AutoConfig, AutoModelForCausalLM, BitsAndBytesConfig
from accelerate.utils import BnbQuantizationConfig, load_and_quantize_model
from accelerate import load_checkpoint_and_dispatch

from datasets import load_dataset
from peft import LoraConfig, PeftModel, get_peft_model
from trl import SFTTrainer

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchsummary import summary
import torch.optim as optim
from torch.optim import AdamW

import torchtext
import torchdata
import spacy
import tqdm
import evaluate
import datasets
from datasets import load_dataset,Value, Sequence, Features, load_from_disk
from datasets import Dataset
from transformers import AutoTokenizer, AutoModel, DataCollatorWithPadding, AutoModelForMaskedLM
from transformers import DataCollatorForLanguageModeling
from transformers import DistilBertTokenizer, DistilBertModel
from transformers import AutoConfig, AutoModelForMaskedLM, BertForMaskedLM, BitsAndBytesConfig

from torch.utils.data import Sampler
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


import sys, psutil, os,copy, math, pickle
import time
import datetime
import random
import logging
logging.raiseExceptions = False
import logging.handlers
from packaging import version

import collections
import unicodedata
import unidecode
import string
import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from IPython import display
display.set_matplotlib_formats('svg')
import plotly.express as px
import plotly.graph_objects as go

"""**Load 4bit quantized llama-2-7b-chat**

- quantization using bitsandbyates
- add new pad token to tokenizer and resize tokenizer embeddings
"""

if torch.cuda.is_bf16_supported():
  compute_dtype = torch.bfloat16
  attn_implementation = 'flash_attention_2'
else:
  compute_dtype = torch.float16
  attn_implementation = 'sdpa'


model_name = "NousResearch/llama-2-7b-chat-hf"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, use_fast=True)


#Create a new token and add it to the tokenizer
tokenizer.add_special_tokens({"pad_token":"<pad>"})
tokenizer.padding_side = 'left'



bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4", # used in qlora . the other option is FP4
    bnb_4bit_compute_dtype=compute_dtype
)

model_4bit = AutoModelForCausalLM.from_pretrained(model_name, device_map='auto',
                                             quantization_config = bnb_config,
                                                  attn_implementation=attn_implementation)

model_4bit.resize_token_embeddings(len(tokenizer))
model_4bit.config.pad_token_id = tokenizer.pad_token_id

"""Load alteon commands dataset

- I have uploaded it to huggingface
https://huggingface.co/spaces/bridge4/


"""

dataset_name = "instructionsetalteonmasterdataset124"
valid_dataset = "instructionsetalteonvalid"


dataset = load_from_disk(dataset_name)
validdataset = load_from_disk(valid_dataset)

"""**Set up PEFT, LORA and SFT paramters for training**"""

model_4bit.config.pretraining_tp = 1

model_4bit.config.use_cache = False # The cache is only used for generation, not for training.

lora_r = 256 # the dimension of the low-rank matrices

lora_alpha = 512 # the scaling factor for the low-rank matrices

lora_dropout = 0.1 # Dropout probability for LoRA layers

########lora config with all linear layers as target modules#################
Peftconfig = LoraConfig(
    r=lora_r,
    lora_alpha=lora_alpha,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','down_proj','up_proj','lm_head'],
    lora_dropout=lora_dropout,
    bias="none", # no biases with be update
    task_type="CAUSAL_LM"
)


# Output directory where the model predictions and checkpoints will be stored
output_dir = "LLMalteon"

# Number of training epochs
num_train_epochs = 1 #max_steps overrides this.

fp16 = True
bf16 = False


per_device_train_batch_size = 1


per_device_eval_batch_size = 1


gradient_accumulation_steps = 4


gradient_checkpointing = True


max_grad_norm = 0.3


learning_rate = 3e-4


weight_decay = 0.01

optim = "paged_adamw_32bit"


lr_scheduler_type = "constant"

# Number of training steps (overrides num_train_epochs)
max_steps = -1

warmup_ratio = 0.03

group_by_length = True

save_steps = 20 #50

logging_steps = 5

max_seq_length = None # picks default from tokenizer of model

#where multiple short examples are packed in the same input sequence to increase training efficiency.
packing = False

device_map ="auto"

######## Some validation specific params #####

evaluation_strategy = 'steps'

per_device_eval_batch_size = 1
eval_accumulation_steps = 1
eval_delay = 0

eval_steps = 5

predict_with_generate = False
generation_max_length = 100

auto_find_batch_size = False

"""**Verify trainable model params after loading PEFT model**"""

def print_trainable_parameters(model):
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param:.2f}"
    )
print_trainable_parameters(get_peft_model(model_4bit, Peftconfig))
# trainable params: 648872192 || all params: 4149293312 || trainable%: 15.64

"""**Training**"""

model_4bit.config.use_cache = False
training_arguments = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=per_device_train_batch_size,
    gradient_accumulation_steps=gradient_accumulation_steps,
    optim=optim,
    save_total_limit = 2,
    save_steps=save_steps,
    logging_steps=logging_steps,
    learning_rate=learning_rate,
    fp16=fp16,
    bf16=bf16,
    max_grad_norm=max_grad_norm,
    max_steps=max_steps,
    warmup_ratio=warmup_ratio,
    group_by_length=group_by_length,
    lr_scheduler_type=lr_scheduler_type,
    num_train_epochs=num_train_epochs,
    evaluation_strategy= evaluation_strategy,
    per_device_eval_batch_size=per_device_eval_batch_size,
    eval_accumulation_steps= eval_accumulation_steps,
    eval_steps= eval_steps,
    report_to="tensorboard"
)

trainer = SFTTrainer(
    model=model_4bit,
    train_dataset=dataset,
    eval_dataset=validdataset,
    peft_config=Peftconfig,
    dataset_text_field="prompt",
    max_seq_length=max_seq_length,
    tokenizer=tokenizer,
    args=training_arguments,
    packing=packing,
)

trainer.train()

"""**Inference**"""

#### merge LORA adapter and unload #######

adapter = "checkpoint-80"
model_qlora = PeftModel.from_pretrained(model_4bit, adapter)
model_qlora = model_qlora.merge_and_unload()

model_qlora.config.use_cache = True

pipe = pipeline(task="text-generation", model=model_qlora, tokenizer=tokenizer,
                max_length=1024, torch_dtype=compute_dtype, device_map="auto")

promptls = ['what are the commands to configure a virt called vgdrf with ip address 10.107.246.1 and service ftp'
,'commands for virtual server named 80 with vip address 10.105.246.110 and service https and group id hy77'
,'what are the commands to configure a real server called 34 and real ip 1.1.1.1'
,'configure real id 34 and ip 192.168.43.4'
,'configure real server called 34 and ip 8.89.98.2'
," commands for alteon virt with id 69769, vip 192.1.2.4, service https and group called ki"
,"what are the commands to configure a virtual server called vamp and with vip 10.5.3.2 and service https and group called bast"
,"can you get me commands for an alteon group called named 90_can that with real servers test1 and test2"
,"what are the commands to configure a virtual server with ip address 10.5.3.2 called 20 with service https and group called bast"]

for prompt in promptls:

  prompt = "### Instruction: "+prompt

  result = pipe(f"<s>[INST] {prompt} [/INST]", use_cache= True)
  print(result[0]['generated_text'])

