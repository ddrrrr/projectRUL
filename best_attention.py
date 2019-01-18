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
    def __init__(self, input_size, hidden_size, cnn_k_s, strides,
                 n_layers=1, dropout=0.5):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cnn_kernel_size = cnn_k_s
        self.cnn_strides = strides
        self.cnn = nn.Sequential(
            nn.Conv1d(self.input_size, 64, self.cnn_kernel_size, self.cnn_strides),
            # nn.ReLU()
            nn.PReLU()
            )
        self.gru = nn.GRU(64, hidden_size, n_layers,
                          dropout=dropout, bidirectional=True)

    def forward(self, x, hidden=None):
        x = x.permute(1,2,0)  # [B*N*T]
        # padding = self.cnn_kernel_size - x.size(2) % self.cnn_strides
        # x = F.pad(x, (0,padding))
        x = self.cnn(x)
        x = x.permute(2,0,1).contiguous()  # [T*B*N]
        outputs, hidden = self.gru(x, hidden)
        outputs = (outputs[:, :, :self.hidden_size] +
                   outputs[:, :, self.hidden_size:])
        return outputs, hidden


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.attn = nn.Sequential(
            nn.Linear(self.hidden_size * 2, hidden_size),
            nn.ReLU()
            )
        self.v = nn.Parameter(torch.rand(hidden_size))
        stdv = 1. / math.sqrt(self.v.size(0))
        self.v.data.uniform_(-stdv, stdv)

    def forward(self, hidden, encoder_outputs):
        timestep = encoder_outputs.size(0)
        h = hidden.repeat(timestep, 1,  1).transpose(0, 1)
        encoder_outputs = encoder_outputs.transpose(0, 1)  # [B*T*H]
        attn_energies = self.score(h, encoder_outputs)
        return F.softmax(attn_energies,dim=1).unsqueeze(1)

    def score(self, hidden, encoder_outputs):
        # [B*T*2H]->[B*T*H]
        energy = self.attn(torch.cat([hidden, encoder_outputs], 2))
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
    def __init__(self, encoder, decoder, teacher_forcing_ratio=0.5):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, src, trg, teacher_forcing_ratio=None, is_analyse=False):
        batch_size = src.size(1)
        max_len = trg.size(0)
        vocab_size = self.decoder.output_size
        outputs = Variable(torch.zeros(max_len, batch_size, vocab_size)).cuda()

        encoder_output, hidden = self.encoder(src)
        hidden = hidden[:self.decoder.n_layers]
        if teacher_forcing_ratio == None:
            teacher_forcing_ratio = self.teacher_forcing_ratio
        is_teacher = random.random() < teacher_forcing_ratio
        output = Variable(trg.data[0,] if is_teacher else outputs[0,]).cuda()
        if is_analyse:
            analyse_data = OrderedDict()
            analyse_data['fea_after_encoder'] = encoder_output.data.cpu().numpy()
            analyse_data['atten'] = []
        for t in range(1, max_len):
            output, hidden, attn_weights = self.decoder(
                    output, hidden, encoder_output)
            outputs[t] = output
            is_teacher = random.random() < teacher_forcing_ratio
            output = Variable(trg.data[t,] if is_teacher else output).cuda()
            if is_analyse:
                analyse_data['atten'].append(attn_weights.data.cpu().numpy())
        if is_analyse:
            analyse_data['atten'] = np.concatenate(analyse_data['atten'],axis=0)
            return outputs, analyse_data
        else:
            return outputs


class RUL():
    def __init__(self):
        self.hidden_size = 200
        self.epochs = 750
        self.lr = 4e-3
        self.gama = 0.7
        self.strides = 5
        self.en_cnn_k_s = 8
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
        self.feature_size = train_data[0].shape[2]

        encoder = Encoder(self.feature_size,self.hidden_size,self.en_cnn_k_s,self.strides,n_layers=1,dropout=0.5)
        decoder = Decoder(self.hidden_size,1,n_layers=1,dropout=0.5)
        seq2seq = Seq2Seq(encoder,decoder).cuda()
        # seq2seq = torch.load('./model/newest_seq2seq')
        seq2seq.teacher_forcing_ratio = 0.3
        optimizer = optim.Adam(seq2seq.parameters(), lr=self.lr)
        # optimizer = optim.SGD(seq2seq.parameters(), lr=self.lr)
        # optimizer = optim.ASGD(seq2seq.parameters(), lr=self.lr)
        # optimizer = optim.RMSprop(seq2seq.parameters(), lr=self.lr)

        log = OrderedDict()
        log['train_loss'] = []
        log['val_loss'] = []
        log['test_loss'] = []
        log['teacher_ratio'] = []
        log['mean_er'] = []
        log['mean_abs_er'] = []
        log['score'] = []
        count = 0
        count2 = 0
        count3 = 0
        e0 = 30
        best_loss = 1
        for e in range(1, self.epochs+1):
            train_loss = self._fit(e, seq2seq, optimizer, train_iter, grad_clip=10.0)
            val_loss = self._evaluate(seq2seq, train_iter)
            test_loss,er = self._evaluate(seq2seq, val_iter, cal_er=True)
            score = self._cal_score(er)
            print("[Epoch:%d][train_loss:%.4e][val_loss:%.4e][test_loss:%.4e][mean_er:%.4e][mean_abs_er:%.4e][score:%.4f]"
                % (e, train_loss, val_loss, test_loss, np.mean(er), np.mean(np.abs(er)), np.mean(score)))
            log['train_loss'].append(float(train_loss))
            log['val_loss'].append(float(val_loss))
            log['test_loss'].append(float(test_loss))
            log['teacher_ratio'].append(seq2seq.teacher_forcing_ratio)
            log['mean_er'].append(float(np.mean(er)))
            log['mean_abs_er'].append(float(np.mean(np.abs(er))))
            log['score'].append(float(np.mean(score)))
            pd.DataFrame(log).to_csv('./model/log.csv',index=False)
            if float(val_loss) == min(log['val_loss']):
                torch.save(seq2seq, './model/seq2seq')
            if (float(test_loss)*11 + float(val_loss)*6)/17 <= best_loss:
                torch.save(seq2seq,'./model/best_seq2seq')
                best_loss = (float(test_loss)*11 + float(val_loss)*6)/17
            # if float(np.mean(np.abs(er))) == min(log['mean_abs_er']):
            #     torch.save(seq2seq,'./model/lowest_test_seq2seq')
            if float(np.mean(score)) == max(log['score']):
                torch.save(seq2seq,'./model/best_score_seq2seq')
            torch.save(seq2seq, './model/newest_seq2seq')

            count2 += 1
            if float(train_loss) <= float(val_loss)*0.2:
                count += 1
            else:
                count = 0
            if count >= 3 or count2 >= 100:
                seq2seq.teacher_forcing_ratio *= self.gama
                count -= 1
                count2 = 0

            # if e == 200:
            #     optimizer = optim.ASGD(seq2seq.parameters(), lr=self.lr)

            # if float(train_loss) < min(log['train_loss']):
            #     count3 += 1
            #     if count3 > 20:
            #         optimizer.param_groups[0]['lr'] *= 0.3
            #         count3 = 0
            # else:
            #     count3 = 0
                
            # optimizer.param_groups[0]['lr'] = (self.lr - (e%e0) * (self.lr-1e-7) / e0)*0.99**e

            # if e % 20 == 0:
            #     self._plot_result(seq2seq, train_iter, val_iter)

    def test(self):
        train_data,train_label = self._preprocess('train')
        train_iter = [[train_data[i],train_label[i]] for i in range(len(train_data))]
        test_data,test_label = self._preprocess('test')
        val_iter = [[test_data[i],test_label[i]] for i in range(len(test_data))]

        seq2seq = torch.load('./model/best_seq2seq')
        self._plot_result(seq2seq, train_iter, val_iter)
    
    def online_test(self):
        test_data,test_label = self._preprocess('test')
        val_iter = [[test_data[i],test_label[i]] for i in range(len(test_data))]

        seq2seq = torch.load('./model/1-2_continue_best_score_seq2seq')
        seq2seq.eval()

        online_analyse = OrderedDict()
        online_analyse['test_label'] = test_label
        online_analyse['test_result'] = []

        with torch.no_grad():
            for [data, label] in val_iter:
                for i in range(5):
                    t_data, t_label = torch.from_numpy(data[round(i*data.shape[0]/5):,].copy()), torch.from_numpy(label[round(i*label.shape[0]/5):,].copy())
                    t_data, t_label = t_data.type(torch.FloatTensor), t_label.type(torch.FloatTensor)
                    t_data = Variable(t_data).cuda()
                    t_label = Variable(t_label).cuda()
                    output = seq2seq(t_data, t_label, teacher_forcing_ratio=0.0)
                    online_analyse['test_result'].append(output.data.cpu().numpy())
        
        sio.savemat('online_test.mat',online_analyse)

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

        seq2seq = torch.load('./model/best_seq2seq')
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
        
    def _evaluate(self, model, val_iter, cal_er=False):
        model.eval()
        total_loss = 0
        er = []
        for [data, label] in val_iter:
            with torch.no_grad():
                data, label = torch.from_numpy(data.copy()), torch.from_numpy(label.copy())
                data, label = data.type(torch.FloatTensor), label.type(torch.FloatTensor)
                data = Variable(data).cuda()
                label = Variable(label).cuda()
                output = model(data, label, teacher_forcing_ratio=0.0)
                if cal_er:
                    label_n = label.data.cpu().numpy().reshape(-1,)
                    output_n = output.data.cpu().numpy().reshape(-1,)
                    x = np.arange(label_n.shape[0]) * self.strides
                    er.append(list(map(lambda x:x[1]/x[0],[np.polyfit(x,label_n,1),np.polyfit(x[5:],output_n[5:],1)])))
            loss = F.mse_loss(output,label)
            # loss = F.l1_loss(output,label)
            total_loss += loss.data  
        if cal_er:
            er = list(map(lambda x:(x[0] - x[1]) / x[0], er))
            return total_loss/len(val_iter), np.array(er)
        else:
            return total_loss / len(val_iter)

    
    def _cal_score(self, er):
        '''
        er: a numpy array
        '''
        return np.exp(np.log(.5)*er*(np.sign(er)*12.5-7.5))


    def _fit(self, e, model, optimizer, train_iter, grad_clip=10.0):
        model.train()
        total_loss = 0
        random.shuffle(train_iter)
        for [data, label] in train_iter:
            random_idx = random.randint(0,round(label.shape[0]*0.3))
            data, label = data[random_idx*self.strides:,], label[random_idx:,]
            data, label = torch.from_numpy(data.copy()), torch.from_numpy(label.copy())
            data, label = data.type(torch.FloatTensor), label.type(torch.FloatTensor)
            data, label = Variable(data).cuda(), Variable(label).cuda()
            optimizer.zero_grad()
            output = model(data, label)
            loss = F.mse_loss(output,label)
            # loss = F.l1_loss(output,label)
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
        fea_type='fre'
        if select == 'train':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.train_bearings})
        elif select == 'test':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.test_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.test_bearings})
        else:
            raise ValueError('wrong selection!')

        for i,x in enumerate(temp_label):
            temp_label[i] = np.arange(temp_data[i].shape[0]) + x
            temp_label[i] = temp_label[i][:,np.newaxis,np.newaxis]
            temp_label[i] = temp_label[i] / np.max(temp_label[i])
            temp_label[i] = temp_label[i][:-self.en_cnn_k_s:self.strides] # when chang 10
        for i,x in enumerate(temp_data):
            temp_data[i] = x[::-1,].transpose(0,2,1)
        
        if fea_type == 'time':
            temp_fun = self._get_time_fea
        elif fea_type == 'fre':
            temp_fun = self._get_fre_fea
        elif fea_type == 'all':
            temp_fun = [self._get_time_fea, self._get_fre_fea]
        else:
            raise ValueError('error selection for features!')

        if isinstance(temp_fun,list):
            r_fea = []
            for func in temp_fun:
                r_fea.append([func(x) for x in temp_data])
            r_fea = [np.concatenate((r_fea[j][i] for j in range(len(r_fea))),axis=2) for i in range(len(r_fea[0]))]
        else:
            r_fea = [temp_fun(x) for x in temp_data]

        if is_analyse:
            if isinstance(temp_fun,list):
                r_fea_no_norm = []
                for func in temp_fun:
                    r_fea_no_norm.append([func(x,is_norm=False) for x in temp_data])
                r_fea_no_norm = [np.concatenate((r_fea[j][i] for j in range(len(r_fea))),axis=2) for i in range(len(r_fea[0]))]
            else:
                r_fea_no_norm = [temp_fun(x,is_norm=False) for x in temp_data]
            return r_fea, r_fea_no_norm, temp_label
        else:
            return r_fea, temp_label
        
        #     r_feature = [self._get_time_fea(x) for x in temp_data]

        # time_feature = [self._get_time_fea(x) for x in temp_data]
        # fre_feature = [self._get_fre_fea(x) for x in temp_data]
        # r_feature = [np.concatenate((time_feature[i],fre_feature[i]),axis=2) for i in range(len(time_feature))]
        # if is_analyse:
        #     time_fea_no_norm = [self._get_time_fea(x,is_norm=False) for x in temp_data]
        #     fre_fea_no_norm = [self._get_fre_fea(x,is_norm=False) for x in temp_data]
        #     r_fea_no_norm = [np.concatenate((time_fea_no_norm[i],fre_fea_no_norm[i]),axis=2) for i in range(len(time_fea_no_norm))]
        #     return r_feature, r_fea_no_norm, temp_label
        # else:
        #     return r_feature, temp_label

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
        # self.feature_size = fea.shape[1]
        if is_norm:
            fea = self._normalize(fea,dim=1)
        fea = fea[:,np.newaxis,:]
        return fea
    
    def _get_fre_fea(self, data, is_norm=True):
        fft_data = np.fft.fft(data,axis=2)/data.shape[2]
        fft_data = (np.abs(fft_data))**2
        fea_list = []
        for i in range(5):
            fea_list.append(np.sum(fft_data[:,:,i*256:(i+1)*256],axis=2,keepdims=True))
        fea = np.concatenate(fea_list,axis=2)
        fea = fea.reshape(-1,fea.shape[1]*fea.shape[2])
        # self.feature_size = fea.shape[1]
        if is_norm:
            fea = self._normalize(fea,dim=1)
        fea = fea[:,np.newaxis,:]
        return fea


    def _normalize(self, data, dim=None):
        if dim == None:
            mmrange = 10.**np.ceil(np.log10(np.max(data) - np.min(data)))
            r_data = (data - np.min(data)) / mmrange
        else:
            # mmrange = 10.**np.ceil(np.log10(np.max(data,axis=dim,keepdims=True) - np.min(data,axis=dim,keepdims=True)))
            mmrange = np.max(data,axis=dim,keepdims=True) - np.min(data,axis=dim,keepdims=True)
            r_data = (data - np.min(data,axis=dim,keepdims=True).repeat(data.shape[dim],axis=dim)) \
                / (mmrange).repeat(data.shape[dim],axis=dim)
        return r_data


if __name__ == '__main__':
    process = RUL()
    # process.train()
    # process.analyse()
    # process.test()
    process.online_test()