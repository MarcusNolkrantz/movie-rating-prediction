#!/usr/bin/env python 

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.utils
import os
import sys, getopt

from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from random import randrange
from dataset import custom_dataset
from torch.utils import data
from model import ordinal_model
from tqdm import tqdm


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

def transform_prediction(pred, device):
   
    sigmoid = nn.Sigmoid()
    norm_preds = sigmoid(pred)

    probs = torch.zeros(pred.shape[0],10).to(device) #Probabilities of each class
    
    prev_values = norm_preds[:,0]
    probs[:,0] = 1 - prev_values

    for i in range(1,10):
      
        if i < 9:
            probs[:,i] = torch.clamp( (prev_values - norm_preds[:,i]), min=0.0)
            prev_values = torch.where(probs[:,i] > 0, norm_preds[:,i], prev_values)
        else:
            probs[:,i] = torch.minimum(norm_preds[:,i-1], prev_values)

    return probs



def train_epoch(model, train_loader, loss, optimizer, scheduler, num_samples, device):

    accum_loss = 0.0
    accum_rmse = 0.0
    batch_count = 0

    model.train()

    print(f'\n-----------Training epoch started--------------\n')

    with tqdm(total=num_samples) as pbar:
        for batch in iter(train_loader):

            pred = model.forward(batch)
            
            labels = torch.add(batch['labels'], -1)
            
            tmp_labels = [[1 if label > i else 0 for label in labels] for i in range(1, 10)]
            binary_labels =  torch.FloatTensor(tmp_labels).to(device)
            binary_labels = torch.permute(binary_labels, (1,0))

            train_losses = [loss(pred[:,i], binary_labels[:,i]) for i in range(0,9)]
            train_loss = sum(loss for loss in train_losses)
            accum_loss += train_loss

            prob_pred = transform_prediction(pred, device)
            accum_rmse += rmse(prob_pred, labels)

            train_loss.backward()

            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            batch_count += 1
            pbar.update(batch['ids'].shape[0])

    return {
        'average_loss': accum_loss.item() / batch_count,
        'average_rmse': accum_rmse.item() / batch_count
    }


def eval_epoch(model, valid_loader, loss, num_samples, device):

    accum_loss = 0.0
    accum_rmse = 0.0
    
    correct_preds = 0
    total_preds = 0
    
    batch_count = 0

    with torch.no_grad():
        model.eval()

        print(f'\n-----------Eval epoch started--------------\n')

        with tqdm(total=num_samples) as pbar:
            for batch in iter(valid_loader):

                pred = model.forward(batch)
                labels = torch.add(batch['labels'], -1)
                
                tmp_labels = [[1 if label > i else 0 for label in labels] for i in range(1, 10)]
                binary_labels =  torch.FloatTensor(tmp_labels).to(device)
                binary_labels = torch.permute(binary_labels, (1,0))

                valid_losses = [loss(pred[:,i], binary_labels[:,i]) for i in range(0,9)]
                valid_loss = sum(loss for loss in valid_losses)
                accum_loss += valid_loss

                prob_pred = transform_prediction(pred, device)
                accum_rmse += rmse(prob_pred, labels)

                top_preds = torch.argmax(input=prob_pred, dim=1)
                correct_preds += torch.sum(top_preds == labels).item()
                total_preds += labels.shape[0]

                batch_count += 1
                pbar.update(batch['ids'].shape[0])

    return {
        'average_loss': accum_loss.item() / batch_count,
        'average_rmse': accum_rmse.item() / batch_count,
        'accuracy': correct_preds / total_preds
    }


def main():

    mode = "train"
    model_name = "ordinal_model_123"

    try:
        argv = sys.argv[1:]
        opts, args = getopt.getopt(argv,"m:n:",["mode=","name="])
    except getopt.GetoptError:
        print('test.py -m <mode> -n <name>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ("-m", "--mode"):
            mode = arg
        elif opt in ("-n", "--name"):
            model_name = arg

    # set seed for deterministic behaviour
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device = ", device)

    # Hyperparameters
    bert_model = 'bert-base-uncased'
    batch_size = 8
    num_epochs = 5
    learning_rate = 2e-5

    tokenizer = BertTokenizer.from_pretrained(bert_model)

    df = pd.read_pickle("./datasets/movies.bz2")
    df = undersample(df, min(df['ratings'].value_counts()))

    train, tmp = train_test_split(df, test_size=0.3)
    test, valid = train_test_split(tmp, test_size=0.5)

    train_set = custom_dataset(train, tokenizer, device)
    valid_set = custom_dataset(valid, tokenizer, device)
    test_set = custom_dataset(test, tokenizer, device)

    train_loader = data.DataLoader(train_set, batch_size=batch_size, shuffle=False, drop_last=True)
    valid_loader = data.DataLoader(valid_set, batch_size=batch_size, shuffle=False, drop_last=True)
    test_loader = data.DataLoader(test_set, batch_size=batch_size, shuffle=False, drop_last=True)

    loss = nn.BCEWithLogitsLoss()

    if mode == "test":
        model_path = "/home/marno874/tdde16-project/models/" + model_name + "/model.pt"
        model = ordinal_model(bert_model).to(device)
        try:
            model.load_state_dict(torch.load(model_path))
        except:
            print("Can't find model with that name. Exiting.")
            return -1

        test_res = eval_epoch(model, test_loader, loss, len(test_set), device)
        print(f'\n-------Test finished-------\n Average rmse = {test_res["average_rmse"]}\n Average loss = {test_res["average_loss"]}\n') 

        return -1

    model = ordinal_model(bert_model).to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate, correct_bias=False)

    scheduler = get_linear_schedule_with_warmup(
            optimizer=optimizer,
            num_warmup_steps=0,
            num_training_steps=len(train_loader)*num_epochs
        )
    
    train_losses = []
    train_rmses = []

    valid_losses = []
    valid_rmses = []

    valid_accuracy = []

    best_loss = 10000

    # Training loop
    for i, epoch in enumerate(range(num_epochs)):

        train_res = train_epoch(model, train_loader, loss, optimizer, scheduler, len(train_set), device)
        valid_res = eval_epoch(model, valid_loader, loss, len(valid_set), device)

        train_losses.append(train_res['average_loss'])
        train_rmses.append(train_res['average_rmse'])

        valid_losses.append(valid_res['average_loss'])
        valid_rmses.append(valid_res['average_rmse'])

        valid_accuracy.append(valid_res['accuracy'])

        print(f'\n-------Epoch finished-------\n Train_res = {train_res}\n Valid_res = {valid_res}\n')   

        if valid_res['average_loss'] < best_loss:
                best_loss = valid_res['average_loss']

                # Create directory for the result
                dir_path = "./models/" + model_name
                if not os.path.exists(dir_path):
                    os.makedirs(dir_path)

                # Save model
                model_path = dir_path + "/model.pt"
                torch.save(model.state_dict(), model_path)

    # Write config to file
    config_path = dir_path + "/config.txt"
    f = open(config_path, "w")
    
    f.write(f'Model type = ordinal classification\n')
    f.write(f'Bert model = {bert_model}\n')
    f.write(f'Batch size = {batch_size}\n')
    f.write(f'Num epochs = {num_epochs}\n')
    f.write(f'Learning rate = {learning_rate}\n')
    f.write(f'Train losses = {train_losses}\n')
    f.write(f'Train RMSE\'s = {train_rmses}\n')
    f.write(f'Valid losses = {valid_losses}\n')
    f.write(f'Valid RMSE\'s = {valid_rmses}\n')
    f.write(f'Valid accuracy\'s = {valid_accuracy}\n')

    f.close()

if __name__ == "__main__":
    main()


