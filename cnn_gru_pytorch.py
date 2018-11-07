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

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        m = OrderedDict()
        m['conv1'] = nn.Conv1d(inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn1'] = nn.BatchNorm1d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv1d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm1d(planes)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual

        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        m  = OrderedDict()
        m['conv1'] = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        m['bn1'] = nn.BatchNorm1d(planes)
        m['relu1'] = nn.ReLU(inplace=True)
        m['conv2'] = nn.Conv1d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        m['bn2'] = nn.BatchNorm1d(planes)
        m['relu2'] = nn.ReLU(inplace=True)
        m['conv3'] = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        m['bn3'] = nn.BatchNorm1d(planes * 4)
        self.group1 = nn.Sequential(m)

        self.relu= nn.Sequential(nn.ReLU(inplace=True))
        self.downsample = downsample

    def forward(self, x):
        if self.downsample is not None:
            residual = self.downsample(x)
        else:
            residual = x

        out = self.group1(x) + residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, feature_size=8):
        self.inplanes = 64
        super(ResNet, self).__init__()

        m = OrderedDict()
        m['conv1'] = nn.Conv1d(2, 64, kernel_size=64, stride=2, padding=31, bias=False)
        m['bn1'] = nn.BatchNorm1d(64)
        m['relu1'] = nn.ReLU(inplace=True)
        m['maxpool'] = nn.MaxPool1d(kernel_size=2, stride=2, padding=0)
        self.group1= nn.Sequential(m)

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=4)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=4)
        self.layer4 = self._make_layer(block, 256, layers[3], stride=4)

        self.avgpool = nn.Sequential(nn.AvgPool1d(10))

        self.group2 = nn.Sequential(
            OrderedDict([
                ('fc', nn.Linear(256 * block.expansion, feature_size))
            ])
        )

        self.output = nn.Linear(feature_size,1)

        # initialization weight of conv2d and bias of BN
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                n = m.kernel_size[0] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.group1(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        feature = self.group2(x)
        x = self.output(feature)

        return x,feature

class CNN(nn.Module):
    def __init__(self, feature_size):
        super(CNN, self).__init__()
        self.cnn_net = nn.Sequential(
            nn.Conv1d(2,32,65,1,32),          #in_shape (2,2560)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(8),                #out_shape (32,320)
            nn.Conv1d(32,32,5,1,2),         #in_shape (32,320)
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.MaxPool1d(4),                #out_shape (32,80)
            nn.Conv1d(32,64,3,1,1),         #in_shape (32,80)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),                #out_shape (64,40)
            nn.Conv1d(64,64,3,1,1),         #in_shape (64,40)
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(2),                #out_shape (64,20)
            nn.Conv1d(64,128,3,1,1),        #in_shape (64,20)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),                #out_shape (128,10)
            nn.Conv1d(128,128,3,1,1),        #in_shape (128,10)
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(2),                #out_shape (128,5)
        )
        self.nn_net = nn.Sequential(
            nn.Dropout(0.25),
            nn.Linear(128*5,256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256,128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Linear(128,feature_size),
            nn.BatchNorm1d(feature_size),
            nn.ReLU()
        )
        self.out = nn.Linear(feature_size,1)
        
    def forward(self, x):
        x = self.cnn_net(x)
        x = x.view(x.size(0), -1)           # flatten the output of conv2 to (batch_size, 64*20)
        x = self.nn_net(x)
        output = self.out(x)
        return output, x    # return x for visualization

class GRU(nn.Module):
    def __init__(self, feature_size):
        super(GRU, self).__init__()
        self.gru_1 = nn.GRU(feature_size,32,2,batch_first=True)          # the input size should be the same with h in _gru_fit
        # self.gru_2 = nn.GRU(32,1,1,batch_first=True)
        self.linear = nn.Linear(32,1)
    def forward(self,x,h):
        x,h = self.gru_1(x,h)
        # x,h = self.gru_2(x,h)
        x = self.linear(x)
        return x,h

class Custom_loss(nn.Module):
    def __init__(self):
        super(Custom_loss, self).__init__()
    def forward(self,pred,tru):
        return torch.mean((pred-tru)**2/(tru+1))

class CNN_GRU():
    def __init__(self):
        self.feature_size = 8
        self.dataset = DataSet.load_dataset(name='phm_data')
        self.train_bearings = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
        self.test_bearings = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                                'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                                'Bearing3_3']
    
    def _build_cnn(self):
        # model = CNN(self.feature_size)
        model = ResNet(Bottleneck, [3, 4, 23, 3])

        weight_p, bias_p = [],[]
        # for name, p in model.named_parameters():
        #     if 'bias' in name:
        #         bias_p += [p]
        #     else:
        #         weight_p += [p]
        #     # 这里的model中每个参数的名字都是系统自动命名的，只要是权值都是带有weight，偏置都带有bias，
        #     # 因此可以通过名字判断属性，这个和tensorflow不同，tensorflow是可以用户自己定义名字的，当然也会系统自己定义。
        # self.cnn_optimizer = torch.optim.Adam([
        #         {'params': weight_p, 'weight_decay':1e-6},
        #         {'params': bias_p, 'weight_decay':0}
        #         ], lr=1e-3)

        self.cnn_optimizer = torch.optim.Adam(model.parameters(), lr=0.001)   # optimize all cnn parameters
        # self.cnn_loss_func = Custom_loss()                      # the target label is not one-hotted
        self.cnn_loss_func = nn.MSELoss()
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def _build_gru(self):
        model = GRU(self.feature_size)
        self.gru_optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
        self.gru_loss_func = nn.MSELoss()
        if torch.cuda.is_available():
            model = model.cuda()
        return model

    def _normalize(self,data):
        r_data = np.zeros_like(data)
        for i in range(r_data.shape[0]):
            r_data[i,] = ((data[i]-np.min(data[i]))/(np.max(data[i])-np.min(data[i]))-0.5)*2
        return r_data

    def _add_noise(self,data,snr=0):
        snr = 10**(snr/10.0)
        for i in range(data.size()[0]):
            xpower = torch.sum(data[i,]**2)/data[i,].numel()
            npower = xpower/snr
            data[i,] += torch.randn(size=data[i,].size())*torch.sqrt(npower)
        return data

    def _c_preprocess(self,select='train',is_random=True):
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
    
    def _g_preprocess(self,model,select):
        if select == 'train':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.train_bearings})
        elif select == 'test':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.test_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.test_bearings})
        else:
            raise ValueError('wrong selection!')

        r_temp_label = []
        r_temp_data = []
        for i,x in enumerate(temp_label):
            t_label = [y for y in range(x,x + temp_data[i].shape[0])]
            t_label.reverse()
            r_temp_label.append(np.array(t_label))
            r_temp_data.append(self._cnn_predict(model,self._normalize(np.transpose(temp_data[i],(0,2,1))))[1])

        r_data = []
        r_label = []
        for i in range(10000):
            bearing_idx = random.randint(0,len(r_temp_data)-1)
            random_bearing = r_temp_data[bearing_idx]
            random_bearing_RUL = r_temp_label[bearing_idx]
            start_idx = random.randint(0,random_bearing.shape[0]-101)
            # end_idx = start_idx + random.randint(50,100)
            end_idx = start_idx + 100
            r_t_data = random_bearing[start_idx:end_idx,]
            if r_t_data.shape[0] < 100:
                r_t_data = np.append(np.zeros((100-r_t_data.shape[0],self.feature_size)),r_t_data,axis=0)
            r_data.append(r_t_data)
            r_label.append(random_bearing_RUL[start_idx:end_idx])

        return np.array(r_data),np.array(r_label)

    def _cnn_fit(self,model,data,label,batch_size,epochs,snr=-4):
        model.train()
        data_loader = dataset_ndarry_pytorch(data,label,batch_size,True)
        print_per_sample = 2000
        for epoch in range(epochs):
            counter_per_epoch = 0
            for i,(x_data,x_label) in enumerate(data_loader):
                x_data = x_data.type(torch.FloatTensor)
                x_label = x_label.type(torch.FloatTensor)
                if snr != None:
                    x_data = self._add_noise(x_data)
                if torch.cuda.is_available():
                    x_data = Variable(x_data).cuda()
                    x_label = Variable(x_label).cuda()
                else:
                    x_data = Variable(x_data)
                    x_label = Variable(x_label)
                # 向前传播
                [out,feature] = model(x_data)
                loss = self.cnn_loss_func(out, x_label)
                # 向后传播
                self.cnn_optimizer.zero_grad()
                loss.backward()
                self.cnn_optimizer.step()
                temp_acc = float(np.mean((out.data.cpu().numpy()-x_label.data.cpu().numpy())**2/(x_label.data.cpu().numpy()+1)))
                if i == 0:
                    p_loss = loss
                    p_acc = temp_acc
                else:
                    p_loss += (loss-p_loss)/(i+1)
                    p_acc += (temp_acc-p_acc)/(i+1)

                if i*batch_size > counter_per_epoch:
                    print('Epoch: ', epoch, '| train loss: %.4f' % p_loss.data.cpu().numpy(), '| test accuracy: %.2f' % p_acc)
                    counter_per_epoch += print_per_sample

            torch.cuda.empty_cache()        #empty useless variable

    def _cnn_predict(self,model,data):
        batch_size = 32
        predict_lable = np.array([])
        model.eval()
        prediction = []
        for i in range(math.ceil(data.shape[0]/batch_size)):
            x_data = data[i*batch_size:min(data.shape[0],(i+1)*batch_size),]
            x_data = torch.from_numpy(x_data)
            x_data = x_data.type(torch.FloatTensor)
            x_data = Variable(x_data).cuda()
            x_prediction = model(x_data)
            if len(prediction) == 0:
                for i,x in enumerate(x_prediction):
                    prediction.append(x_prediction[i].data.cpu().numpy())
            else:
                for i,x in enumerate(prediction):
                    prediction[i] = np.append(x,x_prediction[i].data.cpu().numpy(),axis=0)
            del x_prediction
        return prediction

    def test_cnn(self):
        c_train_data,c_train_label = self._c_preprocess()
        c_train_data = self._normalize(c_train_data)
        self.cnn = self._build_cnn()
        self._cnn_fit(self.cnn,c_train_data,c_train_label,64,80,None)

        torch.save(self.cnn,'./model/cnn')
        self.cnn = torch.load('./model/cnn')
    
        c_test_data,c_test_label = self._c_preprocess('test',False)
        c_test_data = self._normalize(c_test_data)
        [predict_label,_] = self._cnn_predict(self.cnn,c_test_data)
        acc = np.mean(np.square(predict_label-c_test_label)/(c_test_label+1))

        plt.subplot(2,1,1)
        plt.plot(c_test_label)
        plt.scatter([x for x in range(predict_label.shape[0])],predict_label,s=2)
        plt.title(str(acc))

        c_test_data,c_test_label = self._c_preprocess('train',False)
        c_test_data = self._normalize(c_test_data)
        [predict_label,_] = self._cnn_predict(self.cnn,c_test_data)
        acc = np.mean(np.square(predict_label-c_test_label)/(c_test_label+1))

        plt.subplot(2,1,2)
        plt.plot(c_test_label)
        plt.scatter([x for x in range(predict_label.shape[0])],predict_label,s=2)
        plt.title(str(acc))
        plt.savefig('./model/temp.png',dip=900)
        plt.show()

    def _gru_fit(self,model,data,label,batch_size,epochs):
        model.train()
        data_loader = dataset_ndarry_pytorch(data,label,batch_size,True)
        print_per_sample = 2000
        for epoch in range(epochs):
            counter_per_epoch = 0
            for i,(x_data,x_label) in enumerate(data_loader):
                x_data = x_data.type(torch.FloatTensor)
                x_label = x_label.type(torch.FloatTensor)
                h = torch.zeros(2,x_data.size()[0],32)
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
                out = model(x_data,h)[0]
                # out = out[:,-1,:]
                out = out.view(out.shape[0],-1)
                loss = self.gru_loss_func(out, x_label)
                # 向后传播
                self.gru_optimizer.zero_grad()
                loss.backward()
                self.gru_optimizer.step()
                if i == 0:
                    p_loss = loss
                else:
                    p_loss += (loss-p_loss)/(i+1)

                if i*batch_size > counter_per_epoch:
                    accuracy = float(np.mean((out.data.cpu().numpy()-x_label.data.cpu().numpy())**2/(x_label.data.cpu().numpy()+1)))
                    print('Epoch: ', epoch, '| train loss: %.4f' % p_loss.data.cpu().numpy(), '| test accuracy: %.2f' % accuracy)
                    counter_per_epoch += print_per_sample

            torch.cuda.empty_cache()        #empty useless variable

    def test_gru(self):
        model = torch.load('./model/resnet')
        g_train_data,g_train_label = self._g_preprocess(model,'train')                        # data.shape=(10000,100,16), label.shape=(10000,)
        self.gru = self._build_gru()
        self._gru_fit(self.gru,g_train_data,g_train_label,64,100)

        torch.save(self.gru,'./model/gru')

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

if __name__ == '__main__':
    process = CNN_GRU()
    # process.test_cnn()
    process.test_gru()