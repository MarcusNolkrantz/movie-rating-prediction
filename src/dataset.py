import torch
from torch.utils import data

class custom_dataset(data.Dataset):
    def __init__(self, df, tokenizer, device):
        self.df = df
        self.tokenizer = tokenizer
        self.device = device


    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, idx):
        review = self.df['reviews'].iloc[idx]
        encoding = self.tokenizer.encode_plus(
            text=review,
            add_special_tokens=True,
            padding='max_length',
            truncation='longest_first',
            max_length=256,
            return_tensors='pt'
        )
        
        return {
            "ids": encoding.input_ids.to(self.device),
            "mask": encoding.attention_mask.to(self.device),
            "labels": torch.tensor(self.df["ratings"].iloc[idx], dtype=torch.long).to(self.device)
        }
