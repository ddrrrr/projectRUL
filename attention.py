import random
import numpy as np 
from collections import OrderedDict
import math
import matplotlib.pyplot as plt 
import pandas as pd 
import torch
from torch import nn, optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from dataset import DataSet

'''
    TODO: 
    build seq2seq model
    should be normalized?
'''

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.gru = nn.GRU(input_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, x, hidden=None):
        outputs, hidden = self.gru(x, hidden)
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1, 1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.relu(attn_energies).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)),dim=2)
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, hidden_size, output_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(hidden_size + output_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input, last_hidden, encoder_outputs):
        # Get the embedding of the current input word (last output word)
        # embedded = self.embed(input).unsqueeze(0)  # (1,B,N)
        embedded = input.unsqueeze(0)
        # Calculate attention weights and apply to encoder outputs
        attn_weights = self.attention(last_hidden[-1], encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1))  # (B,1,N)
        context = context.transpose(0, 1)  # (1,B,N)
        # Combine embedded input word and attended context, run through RNN
        rnn_input = torch.cat([embedded, context], 2)
        output, hidden = self.gru(rnn_input, last_hidden)
        output = output.squeeze(0)  # (1,B,N) -> (B,N)
        context = context.squeeze(0)
        output = self.out(torch.cat([output, context], 1))
        # output = F.log_softmax(output, dim=1)
        return output, hidden, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, trg, teacher_forcing_ratio=0.5):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        output = Variable(trg.data[0,])  # sos
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            output = Variable(trg.data[t,] if is_teacher else output).cuda()
        return outputs


class RUL():
    def __init__(self):
        self.hidden_size = 32
        self.epochs = 50
        self.lr = 1e-4
        self.dataset = DataSet.load_dataset(name='phm_data')
        self.train_bearings = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
        self.test_bearings = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                                'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                                'Bearing3_3']

    
    def train(self):
        train_data,train_label = self._preprocess('train')
        train_iter = [[train_data[i],train_label[i]] for i in range(len(train_data))]
        test_data,test_label = self._preprocess('test')
        val_iter = [[test_data[i],test_label[i]] for i in range(len(test_data))]

        encoder = Encoder(self.feature_size,self.hidden_size,n_layers=2,dropout=0.5)
        decoder = Decoder(self.hidden_size,1,n_layers=1)
        seq2seq = Seq2Seq(encoder,decoder).cuda()
        optimizer = optim.Adam(seq2seq.parameters(), lr=self.lr)

        log = {}
        log['train_loss'] = []
        log['val_loss'] = []
        for e in range(self.epochs):
            train_loss = self._fit(e, seq2seq, optimizer, train_iter)
            val_loss = self._evaluate(seq2seq, val_iter)
            print("[Epoch:%d][train_loss:%.3f][val_loss:%.3f] "
                % (e, train_loss, val_loss))
            log['train_loss'].append(float(train_loss))
            log['val_loss'].append(float(val_loss))
            pd.DataFrame(log).to_csv('./model/log.csv',index=False)
            if float(val_loss) == min(log['val_loss']):
                torch.save(seq2seq, './model/seq2seq')

            if e % 10 == 0:
                self._plot_result(seq2seq, train_iter, val_iter)
        
    def _evaluate(self, model, val_iter):
        model.eval()
        total_loss = 0
        for [data, label] in val_iter:
            with torch.no_grad():
                data, label = torch.from_numpy(data), torch.from_numpy(label)
                data, label = data.type(torch.FloatTensor), label.type(torch.FloatTensor)
                data = Variable(data).cuda()
                label = Variable(label).cuda()
                output = model(data, label, teacher_forcing_ratio=0.0)
            loss = F.mse_loss(output,label)                    
            total_loss += loss.data
        return total_loss / len(val_iter)


    def _fit(self, e, model, optimizer, train_iter, grad_clip=10.0):
        model.train()
        total_loss = 0
        random.shuffle(train_iter)
        for [data, label] in train_iter:
            data, label = torch.from_numpy(data), torch.from_numpy(label)
            data, label = data.type(torch.FloatTensor), label.type(torch.FloatTensor)
            data, label = Variable(data).cuda(), Variable(label).cuda()
            optimizer.zero_grad()
            output = model(data, label)
            loss = F.mse_loss(output,label)
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.data
        return total_loss / len(train_iter)


    def _plot_result(self, model, train_iter, val_iter):
        model.eval()

        labels = []
        outputs = []
        with torch.no_grad():
            for [data, label] in train_iter:
                data, label = torch.from_numpy(data), torch.from_numpy(label)
                data, label = data.type(torch.FloatTensor), label.type(torch.FloatTensor)
                data = Variable(data).cuda()
                label = Variable(label).cuda()
                output = model(data, label, teacher_forcing_ratio=0.0)
                labels.append(label.data.cpu().numpy())
                outputs.append(output.data.cpu().numpy())
        labels = np.concatenate(tuple(x for x in labels), axis=0)
        outputs = np.concatenate(tuple(x for x in outputs), axis=0)
        labels, outputs = labels.reshape(-1,), outputs.reshape(-1,)
        plt.subplot(2,1,1)
        plt.plot(labels)
        plt.plot(outputs)

        labels = []
        outputs = []
        with torch.no_grad():
            for [data, label] in val_iter:
                data, label = torch.from_numpy(data), torch.from_numpy(label)
                data, label = data.type(torch.FloatTensor), label.type(torch.FloatTensor)
                data = Variable(data).cuda()
                label = Variable(label).cuda()
                output = model(data, label, teacher_forcing_ratio=0.0)
                labels.append(label.data.cpu().numpy())
                outputs.append(output.data.cpu().numpy())
        labels = np.concatenate(tuple(x for x in labels), axis=0)
        outputs = np.concatenate(tuple(x for x in outputs), axis=0)
        labels, outputs = labels.reshape(-1,), outputs.reshape(-1,)
        plt.subplot(2,1,2)
        plt.plot(labels)
        plt.plot(outputs)
        plt.show()

    
    def _preprocess(self, select):
        if select == 'train':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.train_bearings})
        elif select == 'test':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.test_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.test_bearings})
        else:
            raise ValueError('wrong selection!')

        for i,x in enumerate(temp_label):
            temp_label[i] = np.arange(temp_data[i].shape[0])[::-1] + x
            temp_label[i] = temp_label[i][:,np.newaxis,np.newaxis]
            temp_label[i] = temp_label[i] / np.max(temp_label[i])
        for i,x in enumerate(temp_data):
            temp_data[i] = x.transpose(0,2,1)
        time_feature = [self._get_time_fea(x) for x in temp_data]
        return time_feature, temp_label

    def _get_time_fea(self, data):
        fea_dict = OrderedDict()
        fea_dict['mean'] = np.mean(data,axis=2,keepdims=True)
        fea_dict['rms'] = np.sqrt(np.mean(data**2,axis=2,keepdims=True))
        fea_dict['kur'] = np.sum((data-fea_dict['mean'].repeat(data.shape[2],axis=2))**4,axis=2) \
                / (np.var(data,axis=2)**2*data.shape[2])
        fea_dict['kur'] = fea_dict['kur'][:,:,np.newaxis]
        fea_dict['skew'] = np.sum((data-fea_dict['mean'].repeat(data.shape[2],axis=2))**3,axis=2) \
                / (np.var(data,axis=2)**(3/2)*data.shape[2])
        fea_dict['skew'] = fea_dict['skew'][:,:,np.newaxis]
        fea_dict['p2p'] = np.max(data,axis=2,keepdims=True) - np.min(data,axis=2,keepdims=True)
        fea_dict['var'] = np.var(data,axis=2,keepdims=True)
        fea_dict['cre'] = np.max(abs(data),axis=2,keepdims=True) / fea_dict['rms']
        fea_dict['imp'] = np.max(abs(data),axis=2,keepdims=True) \
                / np.mean(abs(data),axis=2,keepdims=True)
        fea_dict['mar'] = np.max(abs(data),axis=2,keepdims=True) \
                / (np.mean((abs(data))**0.5,axis=2,keepdims=True))**2
        fea_dict['sha'] = fea_dict['rms'] / np.mean(abs(data),axis=2,keepdims=True)
        fea_dict['smr'] = (np.mean((abs(data))**0.5,axis=2,keepdims=True))**2
        fea_dict['cle'] = fea_dict['p2p'] / fea_dict['smr']

        fea = np.concatenate(tuple(x for x in fea_dict.values()),axis=2)
        fea = fea.reshape(-1,fea.shape[1]*fea.shape[2])
        self.feature_size = fea.shape[1]
        fea = self._normalize(fea,dim=1)
        fea = fea[:,np.newaxis,:]
        return fea
    
    def _get_fre_fea(self, data):
        pass

    def _normalize(self, data, dim=None):
        if dim == None:
            r_data = (data - np.min(data)) / (np.max(data) - np.min(data))
        else:
            r_data = (data - np.min(data,axis=dim,keepdims=True).repeat(data.shape[dim],axis=dim)) \
                / (np.max(data,axis=dim,keepdims=True) - np.min(data,axis=dim,keepdims=True)).repeat(data.shape[dim],axis=dim)
        return r_data


if __name__ == '__main__':
    process = RUL()
    process.train()