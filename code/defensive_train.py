import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import time
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold, KFold
from tensorflow.keras.preprocessing.sequence import pad_sequences
from functools import reduce

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, RandomSampler, SequentialSampler
from torch.optim.lr_scheduler import ExponentialLR

from transformers import BertTokenizer, BertConfig, BertModel, BertTokenizerFast
from transformers import CTRLModel, CTRLTokenizer, TransfoXLModel, TransfoXLTokenizer, XLNetModel, XLNetTokenizer, XLMModel, XLMTokenizer, DistilBertModel, DistilBertTokenizer, RobertaModel, RobertaTokenizer, AutoModel, AutoTokenizer
from transformers import AdamW, BertForSequenceClassification, get_linear_schedule_with_warmup
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from transformers.models.squeezebert.tokenization_squeezebert_fast import BertTokenizerFast
from transformers.models.gpt2.tokenization_gpt2_fast import GPT2Tokenizer
from transformers.models.openai.tokenization_openai import OpenAIGPTTokenizer

from tqdm import tqdm, trange

import warnings
warnings.filterwarnings("ignore")

import os
os.environ["USE_TORCH"]= "True"

df = pd.read_csv("train.tsv", delimiter='\t', header=None, names=['sentence', 'label'])

sentences = df.sentence.values

sentences = ["[CLS] " + sen + " [SEP]" for sen in sentences]

labels = df.label.values
total_labels = np.array(labels)

tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased",do_lower_case=True)

sentence_tokens = [tokenizer.tokenize(sen) for sen in sentences]

max_len = 128

sentence_ids = [tokenizer.convert_tokens_to_ids(sen) for sen in sentence_tokens]

sentence_ids = pad_sequences(sentence_ids, maxlen=max_len, dtype='long', truncating='post', padding='post')

attention_mask = [[1 if id > 0 else 0 for id in sen] for sen in sentence_ids]

for i in range(len(attention_mask)):
  attention_mask[i].extend(sentence_ids[i])

model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased", num_labels=2)

def get_parameters(model, model_init_lr, multiplier, classifier_lr):
    parameters = []
    lr = model_init_lr
    for layer in range(12, -1, -1):
        layer_params = {
            'params': [p for n, p in model.named_parameters() if f'encoder.layer.{layer}.' in n],
            'lr': lr
        }
        parameters.append(layer_params)
        lr *= multiplier
    classifier_params = {
        'params': [p for n, p in model.named_parameters() if 'layer_norm' in n or 'linear' in n
                   or 'pooling' in n],
        'lr': classifier_lr
    }
    parameters.append(classifier_params)
    return parameters

def accuracy(labels, preds):
    preds = np.argmax(preds, axis=1).flatten()
    labels = labels.flatten()
    acc = np.sum(preds == labels) / len(preds)
    return acc


batch_size = 32
device = torch.device("cuda")
model = model.to(device)
train_loss = []


def skfold(first_list, all_labels, k_num=5):
    sfolder = StratifiedKFold(k_num, random_state=42, shuffle=True)
    for train_index, eval_index in sfolder.split(first_list, all_labels):
        print("TRAIN:", train_index, "EVAL:", eval_index)
        X_train, X_eval = np.array(first_list)[train_index], np.array(first_list)[eval_index]
        y_train, y_eval = np.array(all_labels)[train_index], np.array(all_labels)[eval_index]
        tmp1 = np.hsplit(X_train, 2)
        tmp2 = np.hsplit(X_eval, 2)
        X_train = tmp1[1]
        train_masks = tmp1[0]
        X_eval = tmp2[1]
        eval_masks = tmp2[0]
        X_train = torch.tensor(X_train)
        X_eval = torch.tensor(X_eval)
        y_train = torch.tensor(y_train)
        y_eval = torch.tensor(y_eval)
        train_masks = torch.tensor(train_masks)
        eval_masks = torch.tensor(eval_masks)
        train_dataset = TensorDataset(X_train, train_masks, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=4,
                                      pin_memory=True)
        eval_dataset = TensorDataset(X_eval, eval_masks, y_eval)
        eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        yield train_dataloader, eval_dataloader


def trainx(data, loss_fn, tmp_scheduler):
    model.train()
    running_loss = 0.0
    accumulate_step = 10
    losses = []
    pred_ls = []
    label_ls = []
    i = 0
    for inputs_ids, inputs_masks, inputs_labels in data:
        inputs_ids = inputs_ids.to(device)
        inputs_masks = inputs_masks.to(device)
        inputs_labels = inputs_labels.to(device)

        optimizer.zero_grad()

        outputs = model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks, labels=inputs_labels)

        loss = outputs['loss']

        loss.backward()

        optimizer.step()

        tmp_scheduler.step()
        running_loss += loss.item()

    loss = running_loss / len(data) / batch_size
    return loss


def eval(data):
    model.eval()
    eval_acc = 0.0, 0.0

    for inputs_ids, inputs_masks, inputs_labels in data:
        inputs_ids = inputs_ids.to(device)
        inputs_masks = inputs_masks.to(device)
        inputs_labels = inputs_labels.to(device)

        with torch.no_grad():
            preds = model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks)

        preds = preds['logits'].detach().to('cpu').numpy()
        labels = inputs_labels.to('cpu').numpy()

        eval_acc += accuracy(labels, preds)

    acc = eval_acc / len(data) / batch_size
    return acc


n_splits = 5


EPOCHS = 8




best_accuracy = 0
loss, acc = 0.0, 0.0

for train_data, valid_data in skfold(attention_mask, total_labels, n_splits):
    parameters = get_parameters(model, 2e-5, 0.95, 1e-4)
    optimizer = AdamW(parameters)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0,
                                                num_training_steps=len(train_data) * EPOCHS)
    loss_fn = nn.CrossEntropyLoss().to(device)
    for epoch in range(EPOCHS):
        train_losses = trainx(train_data, loss_fn, scheduler)
        valid_acc = eval(valid_data)

