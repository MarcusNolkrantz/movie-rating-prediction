import torch.nn as nn
import torch
import time
import nvsmi

from transformers import BertModel

class classification_model(nn.Module):

    def __init__(self, bert_model):
        super(classification_model, self).__init__()
        self.embedding_model = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.embedding_model.config.hidden_size, 10)

    def forward(self, batch):

        ids = batch['ids'].squeeze(1)
        mask = batch['mask'].squeeze(1)

        embeddings = self.embedding_model.forward(input_ids=ids, attention_mask=mask, return_dict=False)[1]

        out = self.dropout(embeddings)
        return self.linear(out)  # Softmax is included in cross-entropy loss function


class regression_model(nn.Module):

    def __init__(self, bert_model):
        super(regression_model, self).__init__()
        self.embedding_model = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.embedding_model.config.hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, batch):

        ids = batch['ids'].squeeze(1)
        mask = batch['mask'].squeeze(1)

        embeddings = self.embedding_model.forward(input_ids=ids, attention_mask=mask, return_dict=False)[1]

        out = self.dropout(embeddings)
        out = self.linear(out)

        return self.sigmoid(out)*9


class ordinal_model(nn.Module):

    def __init__(self, bert_model):
        super(ordinal_model, self).__init__()
        self.embedding_model = BertModel.from_pretrained(bert_model)
        self.dropout = nn.Dropout(p=0.1)
        self.linear = nn.Linear(self.embedding_model.config.hidden_size, 9)

    def forward(self, batch):

        ids = batch['ids'].squeeze(1)
        mask = batch['mask'].squeeze(1)

        embeddings = self.embedding_model.forward(input_ids=ids, attention_mask=mask, return_dict=False)[1]

        out = self.dropout(embeddings)
        return self.linear(out)
