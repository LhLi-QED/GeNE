"""
Modified based on landmark-attention (https://github.com/epfml/landmark-attention/blob/main/llama_legacy/run_test.py.)
"""
import argparse

import deepspeed
import torch
import sys
import os
import random
import re
import requests
from tqdm import tqdm
sys.path.append("../")

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['TORCH_USE_CUDA_DSA'] = '1'
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from transformers import AutoConfig, AutoModelForCausalLM, LlamaTokenizer
local_rank = int(os.getenv("LOCAL_RANK", "0"))
world_size = int(os.getenv("WORLD_SIZE", "1"))


def generate_prompt(n_garbage):
    """Generates a text file and inserts an execute line at a random position."""
    n_garbage_prefix = random.randint(0, n_garbage)
    n_garbage_suffix = n_garbage - n_garbage_prefix

    task_description = "There is an important info hidden inside a lot of irrelevant text. Find it and memorize them. I will quiz you about the important information there."
    garbage = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
    garbage_inf = " ".join([garbage] * 1000000)
    assert len(garbage_inf) >= n_garbage
    garbage_prefix = garbage_inf[:n_garbage_prefix]
    garbage_suffix = garbage_inf[:n_garbage_suffix]
    pass_key = random.randint(1, 50000)
    information_line = f"The pass key is {pass_key}. Remember it. {pass_key} is the pass key."
    final_question = "What is the pass key? The pass key is"
    lines = [
        task_description,
        garbage_prefix,
        information_line,
        garbage_suffix,
        final_question
    ]
    return "\n".join(lines), pass_key


def test_model(prompt_text, pass_key, model, tokenizer):
    with torch.no_grad():
        inputs = tokenizer(prompt_text, return_tensors='pt')
        inputs = inputs.to(device=local_rank)
        length = inputs['input_ids'].shape[1]
        o = model.generate(inputs.input_ids, max_new_tokens=10, num_beams=1, do_sample=False, temperature=1.0, use_cache=True, synced_gpus=True)[:, length:]
        response = tokenizer.batch_decode(o)[0]
        assert f"The pass key is {pass_key}" in prompt_text

        try:
            pass_key = int(re.search(r'\d+', response).group())
        except:
            pass_key = response[:20]

    return pass_key



def run_passkey(model, tokenizer):
    n_values = [15000, 30000, 61000, 122500, 245400]
    num_tests = 50

    for n in n_values:

        correct_count = 0

        for i in tqdm(range(num_tests)):
            #print(f"\nRunning test {i + 1}/{num_tests} for n = {n}...")
            prompt_text, pass_key = generate_prompt(n)
            model_output = test_model(prompt_text, pass_key, model, tokenizer)
            if local_rank == 0:
                # print(f"Expected number in the prompt: {pass_key}, output: {model_output}")
                if pass_key == model_output:
                    correct_count += 1
                else:
                    print(model_output)

        if local_rank == 0:
            accuracy = (correct_count / num_tests) * 100
            print(f"Accuracy {accuracy} for n = {n}: {accuracy}%")


def parse_args():
    parser = argparse.ArgumentParser(description="Finetune a transformers model on a causal language modeling task")
    parser.add_argument(
        "--model-path",
        type=str,
        default="../checkpoints/gene-64/checkpoint-300",
    )
    parser.add_argument(
        "--tokenizer-path",
        type=str,
        default="../checkpoints/gene-64/checkpoint-300",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=2357,
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_args()
    random.seed(args.seed)
    model_path = args.model_path
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    config.update({"_attn_implementation_internal": "flash_attention_2"})
    config.rope_scaling['log_scale'] = False
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, torch_dtype=torch.bfloat16, config=config)
    tokenizer = LlamaTokenizer.from_pretrained(args.tokenizer_path)

    ds_engine = deepspeed.init_inference(model=model, tensor_parallel={"tp_size": world_size}, dtype=torch.bfloat16)
    ds_engine.eval()

    run_passkey(ds_engine, tokenizer)
