#!/usr/bin/env python 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.utils
import os

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from random import randrange
from dataset import custom_dataset
from torch.utils import data
from model import custom_model

import sys,logging
#debugging
# logger.info('GPU memory inside loop: %.1f' % (torch.cuda.memory_allocated() // 1024 ** 2))
# logger.info('GPU memory total: %.1f' % ( torch.cuda.get_device_properties(0).total_memory // 1024 ** 2))


def undersample(df, samples_per_class):

    def _remove_entries(df, rating, number):
        df_query = df[df['ratings'] == rating].sample(number)
        df.drop(df_query.index, inplace=True)
        return df

    ratings = df['ratings'].value_counts()

    for r_class, r_count in ratings.iteritems():
        diff = r_count - samples_per_class
        df = _remove_entries(df, r_class, diff)

    return df


def rmse(pred, label):
    top_pred = torch.argmax(input=pred, dim=1)
    se = ((top_pred - label)**2).float()
    mse = torch.mean(se)
    rmse = mse**(1/2)
    return rmse


def train_epoch(model, train_loader, loss, optimizer, scheduler):

    accum_loss = 0.0
    accum_rmse = 0.0
    batch_count = 0

    model.train()
    for batch in iter(train_loader):

        pred = model.forward(batch)
        model.forward(batch)
        labels = torch.add(batch['labels'], -1)

        train_loss = loss(pred, labels)
        accum_loss += train_loss
        accum_rmse += rmse(pred, labels)
        train_loss.backward()

        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        batch_count += 1

    return {
        'average_loss': accum_loss.item() / batch_count,
        'average_rmse': accum_rmse.item() / batch_count
    }


def eval_epoch(model, valid_loader, loss):

    accum_loss = 0.0
    accum_rmse = 0.0
    batch_count = 0

    with torch.no_grad():
        model.eval()
        for batch in iter(valid_loader):
        
            pred = model.forward(batch)
            labels = torch.add(batch['labels'], -1)   

            valid_loss = loss(pred, labels)
            accum_loss += valid_loss
            accum_rmse += rmse(pred, labels)

            batch_count += 1

    return {
        'average_loss': accum_loss.item() / batch_count,
        'average_rmse': accum_rmse.item() / batch_count
    }


def main():

    # Hyperparameters
    bert_model = 'bert-base-uncased'
    batch_size = 2 #8
    num_epochs = 1 #3
    learning_rate = 5e-6

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device = ", device)

    tokenizer = BertTokenizer.from_pretrained(bert_model)

    df = pd.read_pickle("./datasets/movies.bz2")
    df = undersample(df, 2)#min(df['ratings'].value_counts()))

    print(df['ratings'].value_counts())

    train, tmp = train_test_split(df, test_size=0.3)
    test, valid = train_test_split(tmp, test_size=0.5)

    train_set = custom_dataset(train, tokenizer, device)
    valid_set = custom_dataset(valid, tokenizer, device)
    test_set = custom_dataset(test, tokenizer, device)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=True, drop_last=True)

    model = custom_model(bert_model).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)

    scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader)*num_epochs
        )

    loss = nn.CrossEntropyLoss()

    best_epoch = 0
    best_rmse = 10
    
    train_losses = []
    train_rmses = []

    valid_losses = []
    valid_rmses = []


    # Create directory for the result
    model_name = "model1"
    dir_path = "./models/" + model_name
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


    # Training loop
    for i, epoch in enumerate(range(num_epochs)):

        train_res = train_epoch(model, train_loader, loss, optimizer, scheduler)
        valid_res = eval_epoch(model, valid_loader, loss)

        train_losses.append(train_res['average_loss'])
        train_rmses.append(train_res['average_rmse'])

        valid_losses.append(valid_res['average_loss'])
        valid_rmses.append(valid_res['average_rmse'])

        if valid_res['average_rmse'] < best_rmse:
            best_rmse = valid_res['average_rmse']
            best_epoch = i + 1

        print(f'-------Epoch finished-------\n Train_res = {train_res}\n Valid_res = {valid_res}\n')   
    
    # Write config to file
    config_path = dir_path + "/config.txt"
    f = open(config_path, "w")
    
    f.write(f'Bert model = {bert_model}\n')
    f.write(f'Batch size = {batch_size}\n')
    f.write(f'Num epochs = {num_epochs}\n')
    f.write(f'Learning rate = {learning_rate}\n')
    f.write(f'Best model RMSE = {best_rmse} after epoch {best_epoch}\n\n')
    f.write(f'Train losses = {train_losses}\n')
    f.write(f'Train RMSE\'s = {train_rmses}\n')
    f.write(f'Valid losses = {valid_losses}\n')
    f.write(f'Valid RMSE\'s = {valid_rmses}\n')

    f.close()

    model_path = dir_path + "/model.pt"
    torch.save(model.state_dict(), model_path)

    # Test on unseen data
    with torch.no_grad():
        model.eval()
        for batch in iter(test_loader):
        
            pred = model.forward(batch)
            labels = torch.add(batch['labels'], -1)   
            top_pred = torch.argmax(input=pred, dim=1)

            print("\n----------test---------\n")
            print(top_pred)
            print(labels)

            break



if __name__ == "__main__":
    main()


