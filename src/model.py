import torch.nn as nn
import torch

from transformers import BertModel

class custom_model(nn.Module):

    def __init__(self, bert_model):
        super(custom_model, self).__init__()
        self.embedding_model = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(p=0.0)
        self.linear = nn.Linear(self.embedding_model.config.hidden_size, 10)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, batch):
        
        ids = batch['ids'].squeeze(1)
        mask = batch['mask'].squeeze(1)

        _, embeddings = self.embedding_model.forward(input_ids=ids, attention_mask=mask, return_dict=False)
        out = self.dropout(embeddings)
        out = self.linear(out)

        return self.softmax(out)
