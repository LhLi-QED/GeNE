import copy
import json
import torch
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class TrainingDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length):
        self.data = []
        self.data_dir = data_dir
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._get_data()

    def _get_data(self):
        with open(self.data_dir, 'r', encoding='utf-8') as f:
            self.data = f.readlines()

    def __getitem__(self, index) -> T_co:
        text = eval(self.data[index])
        if isinstance(text, dict):
            text = text['text']
        return text

    def __len__(self):
        return len(self.data)


class InstructionDataset(Dataset):
    def __init__(self, data_dir, tokenizer, max_length):
        with open(data_dir, 'r', encoding='utf-8') as f:
            self.data = f.readlines()
        self.data = [json.loads(d) for d in self.data]
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, index) -> T_co:
        sample = self.data[index]
        if isinstance(sample, dict):
            sample = sample['text']
            tokens = self.tokenizer(sample, add_special_tokens=False)
            input_ids = tokens['input_ids']
            attention_mask = tokens['attention_mask']
        else:
            input_ids = [self.tokenizer.bos_token_id]
            attention_mask = [1]
            labels = [-100]
            curr = 1
            for round in sample:
                if curr >= self.max_length:
                    break
                user_tokens = self.tokenizer(round['user'], add_special_tokens=False)['input_ids']
                bot_tokens = self.tokenizer(round['assistant'] + self.tokenizer.eos_token, add_special_tokens=False)['input_ids']
                if curr + len(bot_tokens) + 10 > self.max_length:
                    break
                if curr + len(user_tokens) + len(bot_tokens) + 10 > self.max_length:
                    user_tokens = user_tokens[-self.max_length + len(bot_tokens) + 10 + curr:]
                user_tokens = (self.tokenizer("user: ", add_special_tokens=False)['input_ids'] +
                               user_tokens +
                               self.tokenizer("\n\nassistant: ", add_special_tokens=False)['input_ids'])

                input_ids += user_tokens + bot_tokens
                attention_mask += [1] * (len(user_tokens) + len(bot_tokens))
                labels += [-100] * len(user_tokens) + bot_tokens
                curr += len(user_tokens) + len(bot_tokens)

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": input_ids
        }

    def __len__(self):
        return len(self.data)