import math
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import OrderedDict
from dataset import DataSet
import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm

class Custom_loss(nn.Module):
    def __init__(self):
        super(Custom_loss, self).__init__()
    def forward(self,pred,tru):
        return torch.mean((pred-tru)**2/(tru+1))

def dataset_ndarry_pytorch(data,label,batch_size,shuffle):
    assert data.shape[0] == label.shape[0]
    class CustomDataset(torch.utils.data.Dataset):
        def __init__(self,data,label):
            self.data = data
            self.label = label
        def __getitem__(self, index):
            data, label = self.data[index,], self.label[index,]
            return data, label
        def __len__(self):
            return len(self.data)
    customdataset = CustomDataset(data,label)
    return DataLoader(customdataset,batch_size=batch_size,shuffle=shuffle)

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_input_channel, num_channels, num_block=None, kernel_size=3, dropout=0.1):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        num_block = [1]*len(num_channels) if num_block is None else num_block
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_input_channel if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            for j in range(num_block[i]):
                in_channels = num_channels[i] if j > 0 else in_channels
                layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                        padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.linear = nn.Linear(num_channels[-1],1)
    def forward(self, x):
        x =  self.network(x)
        feature = x[:,:,-1].contiguous().view(x.size(0),-1)
        out = self.linear(feature)
        return out,feature

class GRU(nn.Module):
    def __init__(self,feature_size):
        super(GRU, self).__init__()
        self.encoder = nn.Conv1d(2,16,10,10)
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(16,4,10,10),
            nn.Conv1d(4,2,1)
        )
        self.gru_1 = nn.GRU(16,feature_size,4,batch_first=True)          # the input size should be the same with h in _gru_fit
        self.linear = nn.Linear(feature_size,1)
    def forward(self,x,h):
        x = self.encoder(x)
        restore = self.decoder(x)
        x = x.contiguous().transpose(2,1)
        h = h.contiguous()
        feature,h = self.gru_1(x,h)
        feature = feature[:,-1,:].contiguous().view(feature.size(0),-1)
        x = self.linear(feature)
        return x,feature,restore

class TCN_MODEL():
    def __init__(self):
        self.feature_size = 32
        self.dataset = DataSet.load_dataset(name='phm_data')
        self.train_bearings = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
        self.test_bearings = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                                'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                                'Bearing3_3']

    def _normalize(self,data):
        r_data = np.zeros_like(data)
        for i in range(r_data.shape[0]):
            r_data[i,] = ((data[i]-np.min(data[i]))/(np.max(data[i])-np.min(data[i]))-0.5)*2
        return r_data
    
    def _preprocess(self,select,is_random):
        if select == 'train':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.train_bearings})
        elif select == 'test':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.test_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.test_bearings})
        else:
            raise ValueError('wrong selection!')
        train_data = np.array([])
        train_label = np.array([])
        for i,x in enumerate(temp_label):
            t_label = [y for y in range(round(x),round(x + temp_data[i].shape[0]))]
            t_label.reverse()
            if train_data.size == 0:
                train_data = temp_data[i]
                train_label = np.array(t_label)
            else:
                train_data = np.append(train_data,temp_data[i],axis=0)
                train_label = np.append(train_label,np.array(t_label),axis=0)
        assert train_data.shape[0] == train_label.shape[0]
        if is_random:
            idx = [x for x in range(train_data.shape[0])]
            random.shuffle(idx)
            train_data = train_data[idx]
            train_label = train_label[idx]
        return np.transpose(train_data,(0,2,1)),train_label[:,np.newaxis]

    def _build_model(self):
        # model = TCN(
        #     num_input_channel=2,
        #     num_channels=[32,32,32,32,64,64,64,64,self.feature_size],
        #     num_block=[2,1,1,1,1,1,1,2,1],
        #     kernel_size=3,
        #     dropout=0.1
        #     )
        model = GRU(self.feature_size)
        if torch.cuda.is_available():
            model = model.cuda()
        self.custom_loss = Custom_loss()
        self.mse_loss = nn.MSELoss()
        self.optimizer = optim.Adam(model.parameters(),lr=1e-1)
        return model

    def _fit(self,model,data,label,batch_size,epochs):
        model.train()
        data_loader = dataset_ndarry_pytorch(data,label,batch_size,True)
        print_per_sample = 2000
        for epoch in range(epochs):
            counter_per_epoch = 0
            for i,(x_data,x_label) in enumerate(data_loader):
                x_data = x_data.type(torch.FloatTensor)
                x_label = x_label.type(torch.FloatTensor)
                h = torch.zeros(4,x_data.size()[0],self.feature_size)
                h = h.type(torch.FloatTensor)
                if torch.cuda.is_available():
                    x_data = Variable(x_data).cuda()
                    x_label = Variable(x_label).cuda()
                    h = Variable(h).cuda()
                else:
                    x_data = Variable(x_data)
                    x_label = Variable(x_label)
                    h = Variable(h)
                # 向前传播
                [out,_,restore_x] = model(x_data,h)
                # predict_loss = self.custom_loss(out,x_label)
                predict_loss = self.mse_loss(out,x_label)
                restore_loss = self.mse_loss(x_data,restore_x)
                loss = predict_loss + 1e-2 * restore_loss
                # 向后传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                temp_acc = float(np.mean((out.data.cpu().numpy()-x_label.data.cpu().numpy())**2/(x_label.data.cpu().numpy()+1)))
                if i == 0:
                    p_loss = loss
                    p_acc = temp_acc
                    p_restore_loss = restore_loss
                else:
                    p_loss += (loss-p_loss)/(i+1)
                    p_acc += (temp_acc-p_acc)/(i+1)
                    p_restore_loss += (restore_loss-p_restore_loss)/(i+1)

                if i*batch_size > counter_per_epoch:
                    print('Epoch: ', epoch, '| train loss: %.4f' % p_loss.data.cpu().numpy(), '| test accuracy: %.2f' % p_acc, '| restore loss: %.4f' % p_restore_loss.data.cpu().numpy())
                    counter_per_epoch += print_per_sample

            torch.cuda.empty_cache()        #empty useless variable

    def _predict(self,model,data):
        batch_size = 32
        predict_lable = np.array([])
        model.eval()
        prediction = []
        for i in range(math.ceil(data.shape[0]/batch_size)):
            x_data = data[i*batch_size:min(data.shape[0],(i+1)*batch_size),]
            x_data = torch.from_numpy(x_data)
            x_data = x_data.type(torch.FloatTensor)
            x_data = Variable(x_data).cuda() if torch.cuda.is_available() else Variable(x_data)
            h = torch.zeros(4,x_data.size()[0],self.feature_size)
            h = h.type(torch.FloatTensor)
            h = Variable(h).cuda() if torch.cuda.is_available() else Variable(h)
            x_prediction = model(x_data,h)
            if len(prediction) == 0:
                for i,x in enumerate(x_prediction):
                    prediction.append(x_prediction[i].data.cpu().numpy())
            else:
                for i,x in enumerate(prediction):
                    prediction[i] = np.append(x,x_prediction[i].data.cpu().numpy(),axis=0)
            del x_prediction
        return prediction

    def test(self):
        train_data,train_label = self._preprocess('train',True)
        train_data = self._normalize(train_data)
        self.model = self._build_model()
        self._fit(self.model,train_data,train_label,64,50)

        torch.save(self.model,'./model/tcn')
        model = torch.load('./model/tcn')

        test_data,test_label = self._preprocess('test',False)
        test_data = self._normalize(test_data)
        predict_label = self._predict(model,test_data)[0]
        acc = np.mean(np.square(predict_label-test_label)/(test_label+1))
        plt.subplot(2,1,1)
        plt.plot(test_label)
        plt.scatter([x for x in range(predict_label.shape[0])],predict_label,s=2)
        plt.title(str(acc))

        test_data,test_label = self._preprocess('train',False)
        test_data = self._normalize(test_data)
        predict_label = self._predict(model,test_data)[0]
        acc = np.mean(np.square(predict_label-test_label)/(test_label+1))
        plt.subplot(2,1,2)
        plt.plot(test_label)
        plt.scatter([x for x in range(predict_label.shape[0])],predict_label,s=2)
        plt.title(str(acc))
        plt.show()


if __name__ == '__main__':
    process = TCN_MODEL()
    process.test()