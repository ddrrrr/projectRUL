import random
import numpy as np 
from collections import OrderedDict
import math
import matplotlib.pyplot as plt 
import pandas as pd 
import scipy.io as sio
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
    def __init__(self, input_size, conv_size, hidden_size,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.conv_size = conv_size
        self.hidden_size = hidden_size
        self.cnn_kernel_size = 32
        self.cnn_strides = 5
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_size, self.conv_size, self.cnn_kernel_size, self.cnn_strides),
            # nn.ReLU()
            nn.PReLU()
            )
        self.gru = nn.GRU(self.conv_size, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, x, hidden=None):
        x = x.permute(1,2,0)  # [B*N*T]
        padding = self.cnn_kernel_size - x.size(2) % self.cnn_strides
        x = F.pad(x, (0,padding))
        x = self.cnn(x)
        conv_output = x.permute(2,0,1).contiguous()  # [T*B*N]
        outputs, hidden = self.gru(conv_output, hidden)
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return conv_output, outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Linear(self.hidden_size * 2, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, encoder_outputs, hidden):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1,  1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.relu(attn_energies).transpose(0,1).unsqueeze(2) # T*B*1

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = F.softmax(self.attn(torch.cat([hidden, encoder_outputs], 2)),dim=2)
        energy = energy.transpose(1, 2)  # [B*H*T]
        v = self.v.repeat(encoder_outputs.size(0), 1).unsqueeze(1)  # [B*1*H]
        energy = torch.bmm(v, energy)  # [B*1*T]
        return energy.squeeze(1)  # [B*T]


class Decoder(nn.Module):
    def __init__(self, conv_size, hidden_size,
                 n_layers=1, dropout=0.2):
        super(Decoder, self).__init__()
        self.conv_size = conv_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.attention = Attention(hidden_size)
        self.gru = nn.GRU(conv_size, hidden_size,
                          n_layers, dropout=dropout)
        self.out = nn.Sequential(
            nn.Linear(self.hidden_size, 64),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, conv_output, encoder_output, e_hidden):
        atten_weights = self.attention(encoder_output, e_hidden[-1])  # T*B*1
        context = atten_weights.repeat(1,1,conv_output.size(2)).mul(conv_output)    # T*B*N
        d_output, d_hidden = self.gru(conv_output, None)
        d_hidden = d_hidden[-1,:,:]    # B*H
        output = self.out(d_hidden)
        return atten_weights, output


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x, is_analyse=False):
        conv_output, encoder_output, hidden = self.encoder(x)
        atten, out = self.decoder(conv_output, encoder_output, hidden)
        return out


class RUL():
    def __init__(self):
        self.hidden_size = 200
        self.conv_size = 64
        self.epochs = 400
        self.lr = 1e-2
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

        encoder = Encoder(self.feature_size,self.conv_size,self.hidden_size,n_layers=2,dropout=0.5)
        decoder = Decoder(self.conv_size,self.hidden_size,n_layers=2,dropout=0.5)
        seq2seq = Seq2Seq(encoder,decoder).cuda()
        # seq2seq = torch.load('./model/newest_seq2seq')
        optimizer = optim.Adam(seq2seq.parameters(), lr=self.lr)

        log = OrderedDict()
        log['train_loss'] = []
        log['val_loss'] = []
        log['test_loss'] = []
        count = 0
        for e in range(1, self.epochs+1):
            train_loss = self._fit(e, seq2seq, optimizer, train_iter)
            val_loss = self._evaluate(seq2seq, train_iter)
            test_loss = self._evaluate(seq2seq, val_iter)

            print("[Epoch:%d][train_loss:%.4e][val_loss:%.4e][test_loss:%.4e]"
                    % (e, train_loss, val_loss, test_loss))
            log['train_loss'].append(float(train_loss))
            log['val_loss'].append(float(val_loss))
            log['test_loss'].append(float(test_loss))
            pd.DataFrame(log).to_csv('./model/log.csv',index=False)
            if float(val_loss) == min(log['val_loss']):
                torch.save(seq2seq, './model/atten2')
            else:
                count += 1
            torch.save(seq2seq, './model/newest_atten2')
            if count > 10:
                optimizer.param_groups[0]['lr'] *= 0.8

    def test(self):
        train_data,train_label = self._preprocess('train')
        train_iter = [[train_data[i],train_label[i]] for i in range(len(train_data))]
        test_data,test_label = self._preprocess('test')
        val_iter = [[test_data[i],test_label[i]] for i in range(len(test_data))]

        seq2seq = torch.load('./model/atten2')
        self._plot_result(seq2seq, train_iter, val_iter)

    def analyse(self):
        analyse_data = OrderedDict()
        train_data, train_data_no_norm, train_label = self._preprocess('train',is_analyse=True)
        train_iter = [[train_data[i],train_label[i]] for i in range(len(train_data))]
        test_data, test_data_no_norm, test_label = self._preprocess('test',is_analyse=True)
        val_iter = [[test_data[i],test_label[i]] for i in range(len(test_data))]

        analyse_data['train_data'] = train_data
        analyse_data['train_data_no_norm'] = train_data_no_norm
        analyse_data['train_label'] = train_label
        analyse_data['test_data'] = test_data
        analyse_data['test_data_no_norm'] = test_data_no_norm
        analyse_data['test_label'] = test_label

        seq2seq = torch.load('./model/seq2seq')
        seq2seq.eval()

        analyse_data['train_fea_after_encoder'] = []
        analyse_data['train_atten'] = []
        analyse_data['train_result'] = []

        with torch.no_grad():
            for [data, label] in train_iter:
                data, label = torch.from_numpy(data.copy()), torch.from_numpy(label.copy())
                data, label = data.type(torch.FloatTensor), label.type(torch.FloatTensor)
                data = Variable(data).cuda()
                label = Variable(label).cuda()
                output, temp_analyse_data = seq2seq(data, label, teacher_forcing_ratio=0.0, is_analyse=True)
                analyse_data['train_result'].append(output.data.cpu().numpy())
                analyse_data['train_fea_after_encoder'].append(temp_analyse_data['fea_after_encoder'])
                analyse_data['train_atten'].append(temp_analyse_data['atten'])

        analyse_data['test_fea_after_encoder'] = []
        analyse_data['test_atten'] = []
        analyse_data['test_result'] = []

        with torch.no_grad():
            for [data, label] in val_iter:
                data, label = torch.from_numpy(data.copy()), torch.from_numpy(label.copy())
                data, label = data.type(torch.FloatTensor), label.type(torch.FloatTensor)
                data = Variable(data).cuda()
                label = Variable(label).cuda()
                output, temp_analyse_data = seq2seq(data, label, teacher_forcing_ratio=0.0, is_analyse=True)
                analyse_data['test_result'].append(output.data.cpu().numpy())
                analyse_data['test_fea_after_encoder'].append(temp_analyse_data['fea_after_encoder'])
                analyse_data['test_atten'].append(temp_analyse_data['atten'])

        sio.savemat('analyse_data.mat',analyse_data)

        
    def _custom_loss(self, output, label, size):
        return (size/(1-output)-label)**2

    def _evaluate(self, model, val_iter):
        model.eval()
        total_loss = 0
        for [data, label] in val_iter:
            with torch.no_grad():
                label = label[0:1,]
                data, label = torch.from_numpy(data.copy()), torch.from_numpy(label.copy())
                data, label = data.type(torch.FloatTensor), label.type(torch.FloatTensor)
                data = Variable(data).cuda()
                label = Variable(label).cuda()
                output = model(data)
            # loss = self._custom_loss(output,label,data.size(0))
            loss = F.mse_loss(output, label)
            total_loss += loss.data
        return total_loss / len(val_iter)


    def _fit(self, e, model, optimizer, train_iter, grad_clip=10.0):
        model.train()
        total_loss = 0
        random.shuffle(train_iter)
        for [data, label] in train_iter:
            random_idx = random.randint(0,round(data.shape[0]-350))
            random_len = random.randint(100,data.shape[0] - random_idx - 100)
            data = np.concatenate(tuple(data[i+random_idx:i+random_idx+random_len,] for i in range(64)),axis=1)
            label = np.concatenate(tuple(label[i+random_idx:i+random_idx+1] for i in range(64)),axis=0)
            # data, label = data[random_idx:,], label[random_idx:random_idx+1,]
            data, label = torch.from_numpy(data.copy()), torch.from_numpy(label.copy())
            data, label = data.type(torch.FloatTensor), label.type(torch.FloatTensor)
            data, label = Variable(data).cuda(), Variable(label).cuda()
            optimizer.zero_grad()
            output = model(data)
            # loss = self._custom_loss(output,label,data.size(0))
            loss = F.mse_loss(output, label)
            loss.backward()
            clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            total_loss += loss.data
            torch.cuda.empty_cache()        #empty useless variable
        return total_loss / len(train_iter)


    def _plot_result(self, model, train_iter, val_iter):
        model.eval()

        labels = []
        outputs = []
        with torch.no_grad():
            for [data, label] in train_iter:
                data, label = torch.from_numpy(data.copy()), torch.from_numpy(label.copy())
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
                data, label = torch.from_numpy(data.copy()), torch.from_numpy(label.copy())
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

    
    def _preprocess(self, select, is_analyse=False):
        max_len = 500
        if select == 'train':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.train_bearings})
        elif select == 'test':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.test_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.test_bearings})
        else:
            raise ValueError('wrong selection!')

        for i,x in enumerate(temp_label):
            temp_label[i] = np.arange(temp_data[i].shape[0],dtype=np.float64) + x
            temp_label[i] = temp_label[i][:,np.newaxis]
            temp_label[i] = temp_label[i] / np.max(temp_label[i])
            # temp_label[i] = temp_label[i][::5] # when chang 10
        for i,x in enumerate(temp_data):
            temp_data[i] = x[::-1,].transpose(0,2,1)
        time_feature = [self._get_time_fea(x) for x in temp_data]
        if is_analyse:
            time_feature_no_norm = [self._get_time_fea(x, is_norm=False) for x in temp_data]
            return time_feature, time_feature_no_norm, temp_label
        else:
            return time_feature, temp_label

    def _get_time_fea(self, data, is_norm=True):
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
        if is_norm:
            fea = self._normalize(fea,dim=0)
        fea = fea[:,np.newaxis,:]
        return fea
    
    def _get_fre_fea(self, data):
        pass

    def _normalize(self, data, dim=None):
        if dim == None:
            mmrange = 10.**np.ceil(np.log10(np.max(data) - np.min(data)))
            r_data = (data - np.min(data)) / mmrange
        else:
            mmrange = 10.**np.ceil(np.log10(np.max(data,axis=dim,keepdims=True) - np.min(data,axis=dim,keepdims=True)))
            r_data = (data - np.min(data,axis=dim,keepdims=True).repeat(data.shape[dim],axis=dim)) \
                / (mmrange).repeat(data.shape[dim],axis=dim)
        return r_data


if __name__ == '__main__':
    process = RUL()
    process.train()
    # process.test()
    # process.analyse()