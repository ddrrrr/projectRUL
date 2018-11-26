import random
import numpy as np 
from collections import OrderedDict

from dataset import DataSet

'''
    TODO: 
    build seq2seq model
'''
class Attention():
    def __init__(self):
        self.batch_size = 64
        self.epochs = 50
        self.dataset = DataSet.load_dataset(name='phm_data')
        self.train_bearings = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
        self.test_bearings = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                                'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                                'Bearing3_3']

    def train(self):
        train_data,train_label = self._preprocess('train')
        
    
    def _preprocess(self, select, is_random=True):
        if select == 'train':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.train_bearings})
        elif select == 'test':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.test_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.test_bearings})
        else:
            raise ValueError('wrong selection!')

        new_temp_label = np.concatenate(tuple(np.arange(temp_data[i].shape[0])[::-1] + x \
                                            for i,x in enumerate(temp_label)),axis=0)

        new_temp_data = np.concatenate(tuple(x for x in temp_data),axis=0)
        new_temp_data = new_temp_data.transpose(0,2,1)

        time_feature = self._get_time_fea(new_temp_data)
        if is_random:
            idx = [i for i in range(time_feature.shape[0])]
            random.shuffle(idx)
            time_feature = time_feature[idx]
            new_temp_label = new_temp_label[idx]
        return time_feature, new_temp_label


    def _fit(self, model, data, label):
        pass

    def _predict(self, model, data):
        pass

    def _get_time_fea(self, data):
        fea_dict = OrderedDict()
        fea_dict['mean'] = np.mean(data,axis=2,keepdims=True)
        # fea_dict['mean'] = _normalize(fea_dict['mean'])
        fea_dict['rms'] = np.sqrt(np.mean(data**2,axis=2,keepdims=True))
        # fea_dict['rms'] = _normalize(fea_dict['rms'])
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
        return fea
    
    def _get_fre_fea(self, data):
        pass


if __name__ == '__main__':
    process = Attention()
    process.train()