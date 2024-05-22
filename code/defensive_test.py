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

model_save_path = '/content/drive/MyDrive/Version5Model/model/model.pkl'
tokenizer_save_path = '/content/drive/MyDrive/Version5Model/tokenizer/'

model = torch.load(model_save_path)
tokenizer = BertTokenizerFast.from_pretrained(tokenizer_save_path)

df = pd.read_csv("test.tsv", delimiter='\t', header=None, names=['sentence', 'label'])

sentences = df.sentence.values

labels = df.label.values

sentences = ["[CLS] " + sen + " [SEP]" for sen in sentences]

sentences_tokens = [tokenizer.tokenize(sen) for sen in sentences]

sentence_ids = [tokenizer.convert_tokens_to_ids(sen) for sen in sentences_tokens]

max_len=128

sentence_ids = pad_sequences(sentence_ids, maxlen=max_len, dtype='long', truncating='post', padding='post')

attention_mask = [[1 if id > 0 else 0 for id in sen] for sen in sentence_ids]

sentence_ids = torch.tensor(sentence_ids)

attention_mask = torch.tensor(attention_mask)

labels = torch.tensor(labels)

test_dataset = TensorDataset(sentence_ids, attention_mask, labels)

batch_size=32

test_dataloader = DataLoader(test_dataset, batch_size, shuffle=True)

model.eval()
device = torch.device("cuda")
model.to(device)
test_loss, test_acc = 0.0, 0.0
steps = 0
num = 0

for batch in test_dataloader:

    batch = tuple(data.to(device) for data in batch)

    inputs_ids, inputs_masks, inputs_labels = batch

    with torch.no_grad():
        preds = model(inputs_ids, token_type_ids=None, attention_mask=inputs_masks)

    preds = preds['logits'].detach().to('cpu').numpy()
    inputs_labels = inputs_labels.to('cpu').numpy()

    acc = accuracy(inputs_labels, preds)

    test_acc += acc

    steps += 1

    num += len(inputs_ids)

