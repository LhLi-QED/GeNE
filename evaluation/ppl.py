import argparse
import json
import math
from datetime import datetime
import pickle
import sys
import os
import torch
import random
import deepspeed
import numpy as np
import pandas as pd
import torch.distributed as dist
from torch.nn import CrossEntropyLoss
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co
from torch.utils.data import DistributedSampler, DataLoader

sys.path.append('../')
from transformers import LlamaTokenizer
from utils import *

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class PPLTestDataset(Dataset):
    def __init__(self, tokens, max_length, pad_token_id, window_size=1024, sample_max_length=64 * 1024):
        self.tokens = tokens
        self.data = []
        self.max_length = max_length
        self.window_size = window_size
        self.pad_token_id = pad_token_id

        assert max_length <= sample_max_length
        self.sample_max_length = sample_max_length

        self.sliding_window_split()

    def sliding_window_split(self):
        for tokens in self.tokens:
            if len(tokens) < self.sample_max_length:
                continue
            all_splits = (len(tokens) - self.max_length) // self.window_size + 1
            for i in range(all_splits):
                self.data.append(tokens[i * self.window_size: i * self.window_size + self.max_length])

    def pad_left(self, batch):
        max_length = max([len(item) for item in batch])
        for idx, item in enumerate(batch):
            batch[idx] = [self.pad_token_id] * (max_length - len(item)) + item
        return batch

    def __getitem__(self, index) -> T_co:
        sample = torch.tensor(self.data[index])
        return {
            "input_ids": sample,
            "labels": sample
        }

    def __len__(self):
        return len(self.data)


def batch_func(batch):
    input_ids = [item["input_ids"] for item in batch]
    labels = [item["labels"] for item in batch]
    return {
        "input_ids": torch.stack(input_ids, dim=0),
        "labels": torch.stack(labels, dim=0)
    }


def run(args):
    set_random_seed(1234)
    args.model_path = args.model_path.strip()
    from gene.configuration_llama_gene import LlamaGeneConfig
    from gene.modeling_llama_gene import LlamaForCausalLM
    config = LlamaGeneConfig.from_pretrained(args.model_path)
    config.update({"_attn_implementation_internal": "flash_attention_2"})

    model = LlamaForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        config=config
    )
    ds_config = get_train_ds_config(stage=0)
    ds_config['train_micro_batch_size_per_gpu'] = 1
    ds_config['train_batch_size'] = 1 * torch.distributed.get_world_size()
    ds_config['gradient_accumulation_steps'] = 1
    model, _, _, _ = deepspeed.initialize(
        model=model,
        config=ds_config,
        dist_init_required=True
    )

    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

    model.eval()

    for data_path in args.test_data_path:
        with open(data_path, 'rb') as file:
            tokens = pickle.load(file)
        input_length = [4096, 8192, 16384, 32768, 65536]
        batch_size = [16, 8, 4, 2, 1]
        print_rank_0(f"In {data_path}:", dist.get_rank())
        for il, bsz in zip(input_length, batch_size):
            dataset = PPLTestDataset(tokens, max_length=il, pad_token_id=tokenizer.unk_token_id)
            sampler = DistributedSampler(dataset)
            dataloader = DataLoader(dataset,
                                    collate_fn=batch_func,
                                    sampler=sampler,
                                    batch_size=bsz)
            if il < 65536:
                compute_ppl(config, model, dataloader, args, il)
            else:
                compute_ppl_plt(config, model, dataloader, args, il)


def compute_ppl(config, model, dataloader, args, max_length):
    all_loss = 0.
    print_rank_0(f"total {len(dataloader)}", dist.get_rank())
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            inputs = to_device(inputs, model.device)
            outputs = model(
                **inputs,
                use_cache=False
            )
            all_loss += outputs.loss.detach().item()
            #print_rank_0(f"{outputs.loss}, {inputs['input_ids'].shape}", dist.get_rank())
    dist.barrier()
    all_loss = all_loss / len(dataloader)
    all_loss = torch.tensor(all_loss).to(model.device)
    all_loss = get_all_reduce_mean(all_loss)
    if dist.get_rank() == 0:
        print(f"length={max_length}, ppl={torch.exp(all_loss).item()}")


def compute_ppl_plt(config, model, dataloader, args, max_length):
    all_loss = 0.
    print_rank_0(f"total {len(dataloader)}", dist.get_rank())
    loss_fct = CrossEntropyLoss(reduction='none')
    with torch.no_grad():
        for idx, inputs in enumerate(dataloader):
            input_ids = inputs['input_ids'].to(model.device)
            outputs = model(
                input_ids=input_ids,
                use_cache=False
            )
            shifted_logits = outputs.logits[:, :-1, :].contiguous()
            shifted_labels = input_ids[:, 1:].contiguous()
            shifted_logits = shifted_logits.view(-1, config.vocab_size)
            shifted_labels = shifted_labels.view(-1)
            all_loss += loss_fct(shifted_logits,
                                 shifted_labels).detach().data.to(torch.float32).view(-1, max_length - 1).mean(dim=0)

    dist.barrier()
    all_loss = all_loss / len(dataloader)
    all_loss = get_all_reduce_mean(all_loss)

    if dist.get_rank() == 0:
        all_loss_mean = all_loss.mean()
        print(f"length={max_length}, ppl={torch.exp(all_loss_mean).item()}")


def parse_args():
    parser = argparse.ArgumentParser(description="Compute perplexity of a transformers model on "
                                                 "a causal language modeling task")
    parser.add_argument(
        "--model_path",
        type=str,
        default='../model/results/gene-64-3/checkpoint-300',
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default='../model/llama2_7b',
    )
    parser.add_argument(
        "--test_data_path",
        type=list,
        default=[
            '../data/proofpile_test_sample_64k.pkl',
            '../data/pg19_sample_64k.pkl'
        ],
    )
    parser.add_argument(
        "--local_rank",
        type=int,
        default=-1,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    if args.local_rank == -1:
        device = torch.device("cuda")
    else:
        torch.cuda.set_device(args.local_rank)
        device = torch.device("cuda", args.local_rank)
        dist.init_process_group(backend='nccl')
    run(args)
