import random
import numpy as np 

from dataset import DataSet

'''
    TODO: 
    feature extract in _preprocess
'''
class Process():
    def __init__(self):
        self.batch_size = 64
        self.epochs = 50
        self.dataset = DataSet.load_dataset(name='phm_data')
        self.train_bearings = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
        self.test_bearings = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                                'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                                'Bearing3_3']
    
    def _preprocess(self, select, is_random=True):
        if select == 'train':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.train_bearings})
        elif select == 'test':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.test_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.test_bearings})
        else:
            raise ValueError('wrong selection!')

        new_temp_label = np.array([])
        for i,x in enumerate(temp_label):
            new_temp_label = np.append(new_temp_label, np.arange(temp_data.shape[0])[::-1] + x, axis=0)
        # ndarray with more than 2 dimensions cannot be appended to empty ndarry
        new_temp_data = temp_data[0]
        for i in range(1,len(temp_data)-1):
            new_temp_data = np.append(new_temp_data, temp_data[i], axis=0)

    def _fit(self, model, data, label):
        pass

    def _predict(self, model, data):
        pass

    def train(self):
        pass


if __name__ == '__main__':
    pass