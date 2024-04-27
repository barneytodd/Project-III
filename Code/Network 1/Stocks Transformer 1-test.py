#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
import math
import numpy as np
import matplotlib.pyplot as plt
import random
import os
from os import listdir
from os.path import isfile, join
import pandas as pd
from torchvision.transforms import ToTensor
import random


# In[ ]:


random.seed(42)
torch.manual_seed(42)


# In[ ]:


#Uses same model as in Stocks Transformer 1


# In[ ]:


class Embedding(nn.Module):
    def __init__(self, input_size, embed_dim): 
        super(Embedding, self).__init__()
        
        self.embed_dim = embed_dim
        self.input_size = input_size
        
    def forward(self, x):
        batch_size = x.size(0) 
        C = torch.randn((batch_size, self.input_size, self.embed_dim)) 
        for i in range(len(x)):
            for j in range(len(x[i])):
                C[i][j]*=x[i][j]
        return C 


# In[ ]:


class PositionalEmbedding(nn.Module):
    def __init__(self, max_seq_len, embed_model_dim):
        super(PositionalEmbedding, self).__init__()
        
        self.embed_dim = embed_model_dim
        
        pe = torch.zeros(max_seq_len, self.embed_dim) 
        
        for pos in range(max_seq_len):
            for i in range(0, self.embed_dim, 2):
                pe[pos, i] = math.sin(pos / (10000 ** ((2*i)/self.embed_dim)))
                pe[pos, i+1] = math.cos(pos / (10000 ** ((2 * (i + 1))/self.embed_dim)))
        
        pe = pe.unsqueeze(0) 
        self.register_buffer('pe', pe) 
        
    def forward(self, x):
        x *= math.sqrt(self.embed_dim) 
        seq_len = x.size(1)
        x += torch.autograd.Variable(self.pe[:,:seq_len], requires_grad=False) 
        return x


# In[ ]:


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim=512, n_heads=8):
        super(MultiHeadAttention, self).__init__()
        
        self.embed_dim = embed_dim
        
        self.n_heads = n_heads
        self.single_head_dim = int(self.embed_dim / self.n_heads) 
        
        self.query_matrix = nn.Linear(self.single_head_dim , self.single_head_dim ,bias=False) 
        self.key_matrix = nn.Linear(self.single_head_dim  , self.single_head_dim, bias=False) 
        self.value_matrix = nn.Linear(self.single_head_dim ,self.single_head_dim , bias=False) 
        
        self.out = nn.Linear(self.n_heads*self.single_head_dim ,self.embed_dim) 
        
    def forward(self, key, query, value, mask=None):
        batch_size = key.size(0)
        seq_length = key.size(1)
        seq_length_query = query.size(1)
        
        key = key.view(batch_size, seq_length, self.n_heads, self.single_head_dim) 
        query = query.view(batch_size, seq_length_query, self.n_heads, self.single_head_dim)
        value = value.view(batch_size, seq_length, self.n_heads, self.single_head_dim)
        
        k = self.key_matrix(key).transpose(1, 2) 
        q = self.query_matrix(query).transpose(1, 2)   
        v = self.value_matrix(value).transpose(1, 2) 

        k_adjusted = k.transpose(-1, -2) 

        product = torch.matmul(q, k_adjusted)

        if mask is not None:
            product = product.masked_fill(mask == 0, float("-1e20")) 

        product = product / math.sqrt(self.single_head_dim) 
        scores = F.softmax(product, dim=-1)
        scores = torch.matmul(scores, v) 
        concat = scores.transpose(1,2).contiguous().view(batch_size, seq_length_query, self.single_head_dim*self.n_heads) 

        output = self.out(concat)

        return output


# In[ ]:


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, dropout_value, expansion_factor = 4, n_heads = 8):
        super(TransformerBlock, self).__init__()
        
        self.attention = MultiHeadAttention(embed_dim, n_heads)
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        self.feed_forward = nn.Sequential(
                        nn.Linear(embed_dim, expansion_factor * embed_dim),
                        nn.ReLU(),
                        nn.Linear(expansion_factor * embed_dim, embed_dim)
        )
        
        self.dropout1 = nn.Dropout(dropout_value)
        self.dropout2 = nn.Dropout(dropout_value)
    
    def forward(self, key, query, value):
        attention_out = self.attention(key, query, value)
        attention_residual_out = attention_out + query
        norm1_out = self.dropout1(self.norm1(attention_out)) 
        
        feed_fwd_out = self.feed_forward(norm1_out)
        feed_fwd_residual_out = feed_fwd_out + norm1_out
        norm2_out = self.dropout2(self.norm2(feed_fwd_residual_out))
        
        return norm2_out


# In[ ]:


class TransformerEncoder(nn.Module):
    def __init__(self, seq_len, input_size, embed_dim, num_layers=2, expansion_factor=4, n_heads=8): 
        super(TransformerEncoder, self).__init__()
        
        self.embedding_layer = Embedding(input_size, embed_dim) 
        self.positional_encoder = PositionalEmbedding(seq_len, embed_dim)
        
        self.layers = nn.ModuleList([TransformerBlock(embed_dim, dropout_value, expansion_factor, n_heads) for i in range(num_layers)])
        
    def forward(self, x):
        embed_out = self.embedding_layer(x)
        out = self.positional_encoder(embed_out)
        for layer in self.layers:
            out = layer(out, out, out)

        return out


# In[ ]:


class DecoderBlock(nn.Module):
    def __init__(self, embed_dim, dropout_value, expansion_factor=4, n_heads=8):
        super(DecoderBlock, self).__init__()
        
        self.attention = MultiHeadAttention(embed_dim, n_heads=8)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout_value)
        self.transformer_block = TransformerBlock(embed_dim, dropout_value, expansion_factor, n_heads)
        
    def forward(self, key, x, value, mask):
        attention = self.attention(x, x, x, mask=mask)
        x = self.dropout(self.norm(attention + x))
        out = self.transformer_block(key, x, value)
        
        return out


# In[ ]:


class TransformerDecoder(nn.Module):
    def __init__(self, decoder_input_size, embed_dim, seq_len, target_output_size, dropout_value, num_layers=2, expansion_factor=4, n_heads=8): ###batch size after target output
        super(TransformerDecoder, self).__init__()
        
        self.embedding = Embedding(decoder_input_size, embed_dim) 
        self.position_embedding = PositionalEmbedding(seq_len, embed_dim)
        self.fst_attention = DecoderBlock(embed_dim, dropout_value, expansion_factor=4, n_heads=8)
        self.layers = nn.ModuleList(
            [
                DecoderBlock(embed_dim, dropout_value, expansion_factor=4, n_heads=8) 
                for _ in range(num_layers)
            ]

        )
        self.fc1_out = nn.Linear(embed_dim, 1)
        self.fc2_out = nn.Linear(decoder_input_size, target_output_size)
        self.dropout = nn.Dropout(dropout_value)
        
    def forward(self, x, enc_out, mask):
        x = self.embedding(x) 
        x = self.position_embedding(x)
        x = self.dropout(x)
        
        for layer in self.layers:
            x = layer(enc_out, x, enc_out, mask=None)
        
        out = self.fc1_out(x)
        out = torch.squeeze(out)
        out = self.fc2_out(out)
        out = torch.squeeze(out)
        return out


# In[ ]:


class Transformer(nn.Module):
    def __init__(self, embed_dim, input_size, decoder_input_size, target_output_size, seq_length,num_layers=2, dropout_value=0.2, expansion_factor=4, n_heads=8): 
        super(Transformer, self).__init__()
        
        self.decoder_input_size = decoder_input_size
        
        self.encoder = TransformerEncoder(seq_length, input_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads) 
        self.decoder = TransformerDecoder(decoder_input_size, embed_dim, seq_length, target_output_size, dropout_value, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads)
        
    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(batch_size, 1, trg_len, trg_len) #returns lower triangular matrix
        return trg_mask
    
    def decode(self, src, trg):
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src)
        out_labels = []
        batch_size, seq_len = src.shape[0], src.shape[1]
        
        out = trg
        for i in range(seq_len):
            out = self.decoder(out, enc_out, trg_mask)
            
            out = out[:, -1, :]
            
            out = out.argmax(-1)
            out_labels.append(out.item())
            out = torch.unsqueeze(out, axis=0)
            
        return out_labels
    
    def forward(self, src, trg):
        trg_mask = self.make_trg_mask(trg)
        enc_out = self.encoder(src) 
        
        outputs = self.decoder(trg, enc_out, trg_mask)
        return outputs


# In[ ]:


input_size = 10
decoder_input_size = 9
target_output_size = 1
num_layers = 6
seq_length = 10 
batch_size = 16
dropout_value = 0.2
num_training_stocks = 5


model = Transformer(embed_dim=32, input_size=input_size, 
                    decoder_input_size=decoder_input_size, target_output_size=target_output_size, seq_length=seq_length,
                    num_layers=num_layers, dropout_value=dropout_value, expansion_factor=4, n_heads=8) 

current_path = os.path.join('')

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.0001)


# In[ ]:


class CustomDataset():
    def __init__(self, inputs, labels, transform=None, target_transform=None):
        self.labels = labels
        self.inputs = inputs
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        inpt = self.inputs[idx]
        label = self.labels[idx]
        if self.transform:
            inpt = self.transform(inpt)
        if self.target_transform:
            label = self.target_transform(label)
        return inpt, label


# In[ ]:


#load same data as in Stocks Transformer
mypath = os.path.join(current_path, "..", "..", "Data", "Network 1 data", 'Stocks')
stocks = [f for f in listdir(mypath) if isfile(join(mypath, f))]

Xtr, Ytr = [], []
Xval, Yval = [], []
Xte, Yte = [], []

block_size = input_size


def build_dataset(prices):
            X, Y = [], []
            if len(prices)<block_size+1:
                return [], []
            for i in range(len(prices)-block_size):
                X.append(prices[i:i+block_size])
                Y.append(prices[i+block_size])
            X = torch.tensor(X)
            Y = torch.tensor(Y)
            return X, Y

for stock in stocks[1:num_training_stocks+1]:
    data = open(os.path.join(mypath, stock)).read().splitlines()[1:]

    year_lookup = {}
    for day in data:
        lst = day.split(',')
        year = int(lst[0][:4])
        closing = float(lst[4])
        if year not in year_lookup.keys():
            year_lookup[year] = [closing]
        else:
            year_lookup[year].append(closing)
    
    prices = []
    train_prices = []
    validation_prices = []
    test_prices = []
    mean = 0
    std = 0
    years = list(year_lookup.keys())
    if stock == stocks[num_training_stocks]:
        for year in years:
            test_prices += year_lookup[year]
            prices += year_lookup[year]
    else:
        training_cut_off = years[math.ceil(0.75*len(years))-1]
        for year in range(years[0], training_cut_off):
            train_prices+=year_lookup[year]
            prices += year_lookup[year]

        for year in range(training_cut_off, years[-1]):
            validation_prices+=year_lookup[year]
            prices += year_lookup[year]

    Xtr += build_dataset(train_prices)[0]
    Ytr += build_dataset(train_prices)[1]
    Xval += build_dataset(validation_prices)[0]
    Yval += build_dataset(validation_prices)[1]
    Xte, Yte = build_dataset(test_prices)
training_data = CustomDataset(Xtr, Ytr)
val_data = CustomDataset(Xval, Yval)
test_data = CustomDataset(Xte, Yte)


# In[ ]:


#find the best checkpoint using validation data
best_checkpoint = 0
losses = {}
scores = {}
for c in range(1, 151):
    print(c)
    with torch.no_grad():
        model.eval()
        checkpoint = 10*c
        state_dict = torch.load(os.path.join(current_path, "Checkpoints", "Without Smoothing", '1checkpoint' + str(checkpoint) + '.pth'))
        model.load_state_dict(state_dict)
        inpt = [x for x in val_data[0][0]]
        out = [x for x in val_data[0][0]]
        for i in range(len(val_data)-10):
            x = val_data[i][0]
            y = val_data[i][1]
            x = x.unsqueeze(0)
            trg = x[:, 1:]
            optimizer.zero_grad()
            output = model(x, trg)
            out.append(output)
            inpt.append(val_data[i+1][1])



    total_loss = 0.0
    expected_loss = 0.0
    for i in range(len(inpt)-5):
        outpt = out[i+5].unsqueeze(0)
        total_loss += (inpt[i+5]-out[i+5])**2
        expected_loss += (inpt[i+5]-inpt[i+4])**2
    losses[checkpoint] = total_loss/(len(inpt)-10)
    
    prod1 = 1
    for i in range(1, len(inpt)):
        sign = 1
        if out[i]-out[i-1] < 0:
            sign = -1
        prod1 *= (1 + ((inpt[i]-inpt[i-1])/inpt[i-1]*sign))
    scores[checkpoint] = prod1
    
sorted_losses = dict(sorted(losses.items(), key=lambda item: item[1]))
sorted_scores = dict(sorted(scores.items(), key=lambda item: item[1]), reverse=True)

good_checkpoints = []
threshold = min(losses.values()) + 2

for check in losses.keys():
    if losses[check] < threshold:
        good_checkpoints.append(check)

for check in sorted_scores.keys():
    if check in good_checkpoints:
        best_checkpoint = check
        break

print(best_checkpoint)


# In[ ]:


#test on aal.us
checkpoint = 810
state_dict = torch.load(os.path.join(current_path, "Checkpoints", "Without Smoothing", '1checkpoint' + str(checkpoint) + '.pth'))
model.load_state_dict(state_dict)
with torch.no_grad():
    model.eval()
    inpt = [x for x in test_data[0][0]]
    out = [x for x in test_data[0][0]]
    for i in range(len(test_data)-10):
        x = test_data[i][0]
        y = test_data[i][1]
        x = x.unsqueeze(0)
        trg = x[:, 1:]
        optimizer.zero_grad()
        output = model(x, trg)
        out.append(output)
        inpt.append(test_data[i+1][1])

plt.plot(inpt, label='real data')
plt.plot(out, alpha=0.5, label='predicted values')
plt.xlabel('Days', fontsize=15)
plt.ylabel('Closing Price', fontsize=15)
plt.legend()
plt.show()


# In[ ]:


#choose 20 stocks with low volatility and similar price ranges
good_stocks = []
for stock in stocks:
    data = open(os.path.join(mypath, stock)).read().splitlines()[1:]
    year_lookup = {}
    prices = []
    
    for day in data:
        lst = day.split(',')
        year = int(lst[0][:4])
        closing = float(lst[4])
        if year not in year_lookup.keys():
            year_lookup[year] = [closing]
        else:
            year_lookup[year].append(closing)

    years = list(year_lookup.keys())
    for year in years[-5:]:
        prices+=year_lookup[year]
    
    if len(prices) < 1000:
        continue
    
    max1 = max(prices)
    min1 = min(prices)
    
    grads = []
    for i in range(10, len(prices)):
        grads.append(abs(prices[i] - prices[i-10])/prices[i])
    
    maxgrad = max(grads)
        
    if maxgrad<0.17 and max1<100 and min1>20:
        good_stocks.append(stock)

print(len(good_stocks))


# In[ ]:


#build a set of test data from the above stocks
test_datas = []
for stock in good_stocks:
    test_prices = []
    data = open(os.path.join(mypath, stock)).read().splitlines()[1:]

    year_lookup = {}
    for day in data:
        lst = day.split(',')
        year = int(lst[0][:4])
        closing = float(lst[4])
        if year not in year_lookup.keys():
            year_lookup[year] = [closing]
        else:
            year_lookup[year].append(closing)
    
    years = list(year_lookup.keys())
    for year in years[-3:]:
        test_prices += year_lookup[year]
    Xte, Yte = build_dataset(test_prices)
    test_datas.append(CustomDataset(Xte, Yte))


# In[ ]:


#test network on the above test data
checkpoint = 810
state_dict = torch.load(os.path.join(current_path, "Checkpoints", "Without Smoothing", '1checkpoint' + str(checkpoint) + '.pth'))
model.load_state_dict(state_dict)
prod2 = 1
prods = []
mses = []
for s in range(len(test_datas)):
    test_data = test_datas[s]
    with torch.no_grad():
        model.eval()
        inpt = [x for x in test_data[0][0]]
        out = [x for x in test_data[0][0]]
        for i in range(len(test_data)-10):
            x = test_data[i][0]
            y = test_data[i][1]
            x = x.unsqueeze(0)
            trg = x[:, 1:]
            optimizer.zero_grad()
            output = model(x, trg)
            out.append(output)
            inpt.append(test_data[i+1][1])



    total_loss = 0.0
    for i in range(len(inpt)-10):
        total_loss += (inpt[i+10]-out[i+10])**2
    mse = total_loss/(len(inpt)-10)

    prod1 = 1
    for i in range(1, len(inpt)):
        sign = 1
        if out[i]-out[i-1] < 0:
            sign = -1
        prod1 *= (1+(inpt[i]-inpt[i-1])/inpt[i-1]*sign)
        prod2 *= (1+(inpt[i]-inpt[i-1])/inpt[i-1]*sign)
    score = prod1
    
    prods.append(score)
    mses.append(mse)
    print(checkpoint, mse, score)

    plt.plot(inpt, label='real data')
    plt.plot(out, alpha=0.5, label='predicted values')
    plt.xlabel('Days')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()
    
print(prods, np.mean(prods), mses, np.mean(mses))

