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
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import time
from scipy.interpolate import LSQUnivariateSpline


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
        self.fc1_out = nn.Linear(embed_dim, target_output_size)
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
    def __init__(self, embed_dim, input_size, decoder_input_size, target_output_size, seq_length,num_layers=2, dropout_value=0.2, expansion_factor=4, n_heads=8): ###batch_size, after target output
        super(Transformer, self).__init__()
        
        self.decoder_input_size = decoder_input_size
        
        self.encoder = TransformerEncoder(seq_length, input_size, embed_dim, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads) ###batch_size=batch_size, after embed dim
        self.decoder = TransformerDecoder(decoder_input_size, embed_dim, seq_length, target_output_size, dropout_value, num_layers=num_layers, expansion_factor=expansion_factor, n_heads=n_heads) ###batch_size=batch_size, after target output
        
    def make_trg_mask(self, trg):
        batch_size, trg_len = trg.shape
        trg_mask = torch.tril(torch.ones((trg_len, trg_len))).expand(batch_size, 1, trg_len, trg_len) 
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
num_training_stocks = 4


model = Transformer(embed_dim=32, input_size=input_size, 
                    decoder_input_size=decoder_input_size, target_output_size=target_output_size, seq_length=seq_length,
                    num_layers=num_layers, dropout_value=dropout_value, expansion_factor=4, n_heads=8) ###batch_size=batch_size, after num layers

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


#load data
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

#choose which method of smoothing to use
method = 'mov'
smooth_const = 5
        
#create the datasets
all_train_prices = [0]
all_smooth_prices = [0]
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
    
    #build dataset
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
        
    mean = np.mean(prices)
    std = np.std(prices)
    
    train_prices = (train_prices-mean)/std
    validation_prices = (test_prices-mean)/std
    test_prices = (test_prices-mean)/std
    
    #moving average
    if method == 'mov':
        smooth_prices = [price for price in train_prices[:smooth_const-1]]
        for i in range(smooth_const, len(train_prices)+1):
            smooth_prices.append(sum(train_prices[i-smooth_const:i])/smooth_const)
        

    #exponential moving average
    if method == 'exp':
        if len(train_prices) > 0:
            smooth_prices = [train_prices[0]]
            for i in range(1, len(train_prices)):
                smooth_prices.append(smooth_const*train_prices[i]+(1-smooth_const)*smooth_prices[i-1])
            
    
    
    #adjusted moving average
    if method == 'adj':
        smooth_prices = [price for price in train_prices[:smooth_const-1]]
        for i in range(smooth_const, len(train_prices)+1):
            smooth_prices.append(sum(train_prices[i-smooth_const:i])/smooth_const)

        def bias_calc(lst):
            bias = 0
            count = 0
            for j in range(1, len(lst)):
                if (train_prices[j]-lst[j])*(train_prices[j-1]-lst[j-1]) > 0:
                    count+=1
                    bias+=count**2
                else:
                    count = 0
            return bias

        bias1 = {}
        for alpha in [i/1000 for i in range(31)]:
            bias = [bias_calc(smooth_prices), bias_calc(smooth_prices)]
            Beta = 0
            sign = 1
            end = True
            for step in [0.1, 0.01, 0.001]:
                while bias[1] <= bias[0]:
                    Beta += sign*step
                    adjusted_prices = [price for price in smooth_prices[0:math.ceil(smooth_const*1.5)]]
                    for i in range(math.ceil(smooth_const*1.5) + 1, len(smooth_prices)):
                        if abs((smooth_prices[i]-smooth_prices[i-1])/smooth_prices[i-1]) > alpha:
                            adjusted_prices.append(smooth_prices[i]+Beta*(smooth_prices[i]-smooth_prices[i-1]))
                            end = False
                        else:
                            adjusted_prices.append(smooth_prices[i])
                    bias[0] = bias[1]
                    bias[1] = bias_calc(adjusted_prices)
                    if end:
                        break
                bias[0] = bias[1]
                if end:
                    break
                sign*=-1
            bias1[alpha] = [Beta-0.001, bias[0]]

        best_bias = bias1[0.001][1]
        best_alpha = 0.001
        best_beta = 0
        for alpha in bias1.keys():
            if bias1[alpha][1] < best_bias:
                best_alpha = alpha
                best_beta = bias1[alpha][0]
                best_bias = bias1[alpha][1]

        adj_smooth_prices = [price for price in smooth_prices]
        for i in range(math.ceil(smooth_const*1.5) + 1, len(smooth_prices)):
            if abs((smooth_prices[i]-smooth_prices[i-1])/smooth_prices[i-1]) > alpha:
                adj_smooth_prices[i] += best_beta*(smooth_prices[i]-smooth_prices[i-1])
        smooth_prices = adj_smooth_prices
        
    
    

    #spline smoothing
    if method == 'spline':
        if len(train_prices) > 1:
            num_datapoints = smooth_const
            num_knots = max(0, math.floor(len(train_prices)/num_datapoints-1))
            time = np.arange(len(train_prices))

            knot_indices = np.linspace(num_datapoints, len(time) - 1-num_datapoints, num_knots).astype(int)
            knots = time[knot_indices]

            cs = LSQUnivariateSpline(time, train_prices, k=3, t=knots)
            smooth_prices = cs(time)

    Xtr += build_dataset(smooth_prices)[0]
    Ytr += build_dataset(smooth_prices)[1]
    Xval += build_dataset(validation_prices)[0]
    Yval += build_dataset(validation_prices)[1]
    Xte, Yte = build_dataset(test_prices)
    
    all_train_prices.extend(train_prices)
    all_smooth_prices.extend(smooth_prices)

training_data = CustomDataset(Xtr, Ytr)
val_data = CustomDataset(Xval, Yval)


# In[ ]:


#load smoothed prices into dicts to calculate bias and mse for each method
stocks = [f for f in listdir(mypath) if isfile(join(mypath, f))]

def add_to_dict(key, value, dct):
    if key in dct:
        dct[key] = np.concatenate((dct[key],value))
    else:
        dct[key] = value

all_train_prices = []

all_mov_smooth_prices = {}
all_centmov_smooth_prices = {}
all_exp_smooth_prices = {}
all_spline_smooth_prices = {}

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
    
    #build dataset
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
    
    #moving average
    for sc in [3, 5, 10, 20, 50]:
        smooth_const = sc
        mov_smooth_prices = [price for price in train_prices[:smooth_const-1]]
        for i in range(smooth_const, len(train_prices)+1):
            mov_smooth_prices.append(sum(train_prices[i-smooth_const:i])/smooth_const)
        add_to_dict(sc, mov_smooth_prices, all_mov_smooth_prices)
        
    #centered moving average
    for sc in [3, 5, 10, 20]:
        smooth_const = sc
        centmov_smooth_prices = [price for price in train_prices[:math.ceil(smooth_const/2)-1]]
        for i in range(math.ceil(smooth_const/2), len(train_prices)+1-math.floor(smooth_const/2)):
            centmov_smooth_prices.append(sum(train_prices[i-math.ceil(smooth_const/2):i+math.floor(smooth_const/2)])/smooth_const)
        centmov_smooth_prices.extend(train_prices[-math.floor(smooth_const/2):])
        add_to_dict(sc, centmov_smooth_prices, all_centmov_smooth_prices)  
    
    #exponential moving average
    if len(train_prices) > 0:
        for sc in [0.1, 0.15, 0.2, 0.25]:
            smooth_const = sc
            exp_smooth_prices = [train_prices[0]]
            for i in range(1, len(train_prices)):
                exp_smooth_prices.append(smooth_const*train_prices[i]+(1-smooth_const)*exp_smooth_prices[i-1])
            add_to_dict(sc, exp_smooth_prices, all_exp_smooth_prices)

    #spline smoothing
    if len(train_prices) > 1:
        for n in [5, 10, 15, 20]:
            num_datapoints = n
            num_knots = max(0, math.floor(len(train_prices)/num_datapoints-1))
            time = np.arange(len(train_prices))

            knot_indices = np.linspace(num_datapoints, len(time) - 1-num_datapoints, num_knots).astype(int)
            knots = time[knot_indices]

            cs = LSQUnivariateSpline(time, train_prices, k=3, t=knots)
            spline_smooth_prices = cs(time)
            add_to_dict(n, spline_smooth_prices, all_spline_smooth_prices)

    #all_train_prices.extend(train_prices)
    #all_smooth_prices.extend(smooth_prices)


# In[ ]:


#calculates bias values
def bias_calc(lst):
    bias = 0
    count = 0
    for j in range(1, len(lst)):
        if (all_train_prices[j]-lst[j])*(all_train_prices[j-1]-lst[j-1]) > 0:
            count+=1
            bias+=count**2
        else:
            count = 0
    return bias


# In[ ]:


#creates dict for adjusted moving average 
all_adjmov_smooth_prices = {}
Betas = {}
for key in all_mov_smooth_prices.keys():
    bias1 = {}
    for alpha in [i/1000 for i in range(21)]:
        bias = [bias_calc(all_mov_smooth_prices[key]), bias_calc(all_mov_smooth_prices[key])]
        Beta = 0
        sign = 1
        for step in [0.1, 0.01, 0.001]:
            while bias[1] <= bias[0]:
                Beta += sign*step
                adjusted_prices = [price for price in all_mov_smooth_prices[key][0:math.ceil(key*1.5)]]
                for i in range(math.ceil(key*1.5) + 1, len(list(all_mov_smooth_prices[key]))):
                    if abs(all_mov_smooth_prices[key][i]-all_mov_smooth_prices[key][i-1])/all_mov_smooth_prices[key[i-1]] > alpha:
                        adjusted_prices.append(all_mov_smooth_prices[key][i]+Beta*(all_mov_smooth_prices[key][i]-all_mov_smooth_prices[key][i-1]))
                    else:
                        adjusted_prices.append(all_mov_smooth_prices[key][i])
                bias[0] = bias[1]
                bias[1] = bias_calc(adjusted_prices)
            bias[0] = bias[1]
            sign*=-1
        if alpha not in bias1.keys():
            bias1[alpha] = [Beta-0.001, bias[0]]
        elif bias[0] < bias1[alpha][1]:
            bias1[alpha] = [Beta-0.001, bias[0]]
    
    best_bias = bias1[0.001][1]
    best_alpha = 0.001
    best_beta = 0
    for alpha in bias1.keys():
        if bias1[alpha][1] < best_bias:
            best_alpha = alpha
            best_beta = bias1[alpha][0]
        
    Betas[key] = [best_alpha, best_beta]
    smooth_const = key
    adj_smooth_prices = [price for price in all_mov_smooth_prices[key]]
    for i in range(math.ceil(key*1.5) + 1, len(list(all_mov_smooth_prices[key]))):
        if abs(all_mov_smooth_prices[key][i]-all_mov_smooth_prices[key][i-1])/all_mov_smooth_prices[key[i-1]] > alpha:
            adj_smooth_prices[i] += best_beta*(all_mov_smooth_prices[key][i]-all_mov_smooth_prices[key][i-1])
    add_to_dict(key, adj_smooth_prices, all_adjmov_smooth_prices)
print(Betas)


# In[ ]:


#plots the graph of each smoothing method
dicts = [all_mov_smooth_prices, all_centmov_smooth_prices, all_adjmov_smooth_prices, all_exp_smooth_prices, all_spline_smooth_prices]
names = ["mov", "centmov", "adjmov", "exp", "spline"]

for i in range(len(dicts)):
    smooth = dicts[i]
    for key in smooth.keys():
        print(names[i], key)
        plt.plot(all_train_prices[:1000], alpha=0.7, label='real data')
        plt.plot(smooth[key][:1000], label='smoothed values')
        plt.xlabel('Days', fontsize=15)
        plt.ylabel('Closing Price', fontsize=15)
        plt.legend()
        plt.show()


# In[ ]:


#calculate the MSE for each method
for i in range(len(dicts)):
    smooth = dicts[i]
    for key in smooth.keys():
        print(names[i], key)
        sum1=0
        for j in range(1000):
            sum1 += (all_train_prices[j]-smooth[key][j])**2
        sum1/=len(all_train_prices)
        print(sum1*10000)


# In[ ]:


#calculate the bias values for each method
for i in range(len(dicts)):
    smooth = dicts[i]
    for key in smooth.keys():
        print(names[i], key)
        print(bias_calc(smooth[key]))


# In[ ]:


trainloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
valloader = DataLoader(val_data, batch_size=batch_size, shuffle=True)


# In[ ]:


epochs = 1500
train_losses, val_losses = [], []
for e in range(1,epochs):
    start_time = time.time()
    tot_train_loss = 0
    model.train()
    for x, y in trainloader:
        if len(y) != batch_size:
            continue
        trg = x[:, 1:]
        optimizer.zero_grad()

        outputs = model(x, trg)
        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()

        tot_train_loss += loss.item()

    else:
        tot_val_loss = 0
        with torch.no_grad():
            model.eval()
            for x, y in valloader:
                if len(y) != batch_size:
                    continue
                trg = x[:, 1:]
                outputs = model(x, trg)
                loss = criterion(outputs, y)
                tot_val_loss += loss.item()

        train_loss = tot_train_loss / len(trainloader.dataset)
        val_loss = tot_val_loss / len(valloader.dataset)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        print("Epoch: {}/{}.. ".format(e+1, epochs),
              "Training Loss: {:.3f}.. ".format(train_loss),
              "Test Loss: {:.3f}.. ".format(val_loss), 
              "Time taken: {:.3f}..".format(time.time()-start_time))
        
    if e%10 == 0:
        torch.save(model.state_dict(), 'smoothcheckpoint' + str(e) + '.pth')

