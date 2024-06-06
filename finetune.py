import copy
import os
import random

import yaml
import math
import torch
import pickle
from functools import partial
import argparse


import torch.distributed as dist
from torch.optim import AdamW
from transformers import LlamaTokenizer

from transformers import Trainer, TrainingArguments
from dataset import TrainingDataset, InstructionDataset

from gene.modeling_llama_gene import LlamaForCausalLM
from gene.configuration_llama_gene import LlamaGeneConfig
from utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def post_train_batch_func(batch, tokenizer, max_length, ignored_id=-100):
    batch_size = len(batch)
    o = tokenizer(
        batch,
        truncation='longest_first',
        max_length=max_length,
        padding=True,
        return_tensors='pt',
        add_special_tokens=False,
    )
    input_ids = o['input_ids']
    attention_mask = o['attention_mask']
    labels = copy.deepcopy(input_ids)
    labels[labels == tokenizer.pad_token_id] = ignored_id
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "use_cache": False
    }


def ift_batch_func(batch, tokenizer, max_length, ignored_id=-100):
    def pad_right(ids, max_length, pad_id):
        return ids + [pad_id] * (max_length - len(ids))

    max_len = max([len(item['input_ids']) for item in batch])
    if max_len > max_length:
        print(max_len)
    input_ids = [pad_right(item['input_ids'], max_len, tokenizer.pad_token_id) for item in batch]
    attention_mask = [pad_right(item['attention_mask'], max_len, 1) for item in batch]
    labels = [pad_right(item['labels'], max_len, ignored_id) for item in batch]
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "use_cache": False
    }


def train(args):

    with open(args.training_config_path, "r") as fp:
        train_config = yaml.safe_load(fp)

    task = train_config['task']
    if task == 'ptrain':
        batch_func = post_train_batch_func
        DatasetClass = TrainingDataset
    else:
        batch_func = ift_batch_func
        DatasetClass = InstructionDataset

    ds_config = get_train_ds_config()

    training_args = TrainingArguments(deepspeed=ds_config, **train_config['train'])

    set_random_seed(train_config['train']['seed'])
    train_data_path = train_config['data']['train_data_path']
    max_length = train_config['data'].get('max_length', 16384)

    model_path = train_config['model']['model_name_or_path']
    tokenizer_path = train_config['model']['tokenizer_name_or_path']

    tokenizer = LlamaTokenizer.from_pretrained(tokenizer_path)
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = 'right'

    config = LlamaGeneConfig.from_pretrained(model_path)
    config.update({"_attn_implementation_internal": "flash_attention_2"})
    config.update({
        "rope_scaling": train_config['gene_config']
    })

    train_dataset = DatasetClass(train_data_path, tokenizer, max_length)

    model = LlamaForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        config=config
    )
    data_collator = partial(batch_func, tokenizer=tokenizer, max_length=max_length)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )
    trainer.train()


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--training_config_path",
        type=str,
        default='./configs/ift_16k_64k.yaml',
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
    )
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    torch.distributed.init_process_group("nccl")
    args.device = torch.device("cuda", dist.get_rank())
    train(args)

