# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:53 2018

@author: a273
TODO
"""

import os
import operator
import random
import scipy.io as sio
import pickle as pickle
import numpy as np
import pandas as pd

class DataSet(object):
    '''
        index:list
        dataset:list
    '''
    def __init__(self,name='',index=[],save_path='./data/',dataset=[]):
        self.name = name
        self.index = index
        self.save_path = save_path
        self.dataset = dataset

    # inner function
    def _deal_condition(self,condition):
        conforming_idx_all = []
        for k in condition.keys():
            value_attribute = self.get_value_attribute(k)
            conforming_idx_all.append([x in condition[k] for x in value_attribute])
        conforming_idx = [True]*len(self.dataset)
        for x in conforming_idx_all:
            conforming_idx = conforming_idx and x
        return conforming_idx

    # modify
    def reset_index(self,index):
        assert isinstance(index,list)
        self.index = index

    def add_index(self,new_attribute,new_value=None):
        self.index.append(new_attribute)
        if new_value == None:
            for x in self.dataset:
                x.append(new_value)
        elif isinstance(new_value,list):
            assert len(new_value) == len(self.dataset)
            for i,x in enumerate(self.dataset):
                x.append(new_value[i])
        else:
            raise TypeError

    def del_index(self,del_attribute):
        idx = self.index.index(del_attribute)
        for x in self.dataset:
            del(x[idx])
        self.index.remove(del_attribute)

    def append(self,append_data):
        if isinstance(append_data,dict):
            if len(append_data.keys()) <= len(self.index):
                append_data_list = []
                for x in self.index:
                    if x in list(append_data.keys()):
                        append_data_list.append(append_data[x])
                    else:
                        append_data_list.append(None)
                self.dataset.append(append_data_list)
            else:
                raise ValueError('append_data has too much attribute!')
        elif isinstance(append_data,list):
            if len(append_data) == len(self.index):
                self.dataset.append(append_data)
            else:
                raise ValueError('append_data has wrong number of attribute!')
        else:
            raise TypeError('append_data should be dict or list')

    def delete(self,condition):
        conforming_idx = self._deal_condition(condition)
        for i,x in enumerate(self.dataset):
            if conforming_idx[i]:
                self.dataset.pop(x)

    # get information or values
    def get_value_attribute(self,attribute):
        idx = self.index.index(attribute)
        return [x[idx] for x in self.dataset]

    def get_value(self,attribute,condition={}):
        conforming_idx = self._deal_condition(condition)
        idx = self.index.index(attribute)
        return [x[idx] for i,x in enumerate(self.dataset) if conforming_idx[i]]

    def get_dataset(self,condition={}):
        conforming_idx = self._deal_condition(condition)
        return DataSet(name='temp',index=self.index,
                        dataset=[x for i,x in enumerate(self.dataset) if conforming_idx[i]])

    def get_random_choice(self):
        r = {}
        data = random.choice(self.dataset)
        for i,k in enumerate(self.index):
            r[k] = data[i]
        return r

    def get_random_samples(self,n=1):
        return DataSet(name='temp',index=self.index,dataset=random.sample(self.dataset,n))
    
    # value process
    def normalization(self,attribute,select='std'):
        idx = self.index.index(attribute)
        for i in range(len(self.dataset)):
            if select == 'fft':
                self.dataset[i][idx] = self.dataset[i][idx] / np.max(self.dataset[i][idx])
            else:
                self.dataset[i][idx] = self.dataset[i][idx] - np.mean(self.dataset[i][idx])
                if select == 'min-max':
                    self.dataset[i][idx] = self.dataset[i][idx] / max(np.max(self.dataset[i][idx]),abs(np.min(self.dataset[i][idx])))
                elif select == 'std':
                    self.dataset[i][idx] = self.dataset[i][idx] / np.std(self.dataset[i][idx])
                else:
                    raise ValueError

    # class operation
    def shuffle(self):
        random.shuffle(self.dataset)

    def random_sample(self,n):
        if isinstance(n,str):
            if n == 'all':
                self.shuffle()
            elif n == 'half':
                self.dataset = random.sample(self.dataset,int(len(self.dataset)/2))
            else:
                raise ValueError('n should be \'all\' or \'half\'!')
        elif isinstance(n,int):
            if n >= len(self.dataset):
                self.shuffle()
            else:
                self.dataset = random.sample(self.dataset,n)
        else:
            raise TypeError('n should be int of string!')

    def dataset_filter(self,condition={}):
        conforming_idx = self._deal_condition(condition)
        self.dataset = [x for i,x in enumerate(self.dataset) if conforming_idx[i]]

    def save(self):
        assert self.name != ''
        assert self.save_path != ''
        pickle.dump(self, open(self.save_path + 'DataSet_' +
                                     self.name + '.pkl', 'wb'), True)
        print('dataset ', self.name, ' has benn saved\n')

    def load(self,name=''):
        if name != '':
            self.name = name
        assert self.name != ''
        assert self.save_path != ''
        full_name = self.save_path + 'DataSet_' + self.name + '.pkl'
        load_class = pickle.load(open(full_name, 'rb'))
        assert load_class.name == self.name
        assert load_class.save_path == self.save_path
        print('dataset ', self.name, ' has been load')
        self.dataset = load_class.dataset
        self.index = load_class.index

    @staticmethod
    def load_dataset(name):
        save_path = './data/'
        full_name = save_path + 'DataSet_' + name + '.pkl'
        load_class = pickle.load(open(full_name,'rb'))
        print('dataset ', name, ' has been load')
        return load_class

def make_phm_dataset():
    RUL_dict = {'Bearing1_1':0,'Bearing1_2':0,
                'Bearing2_1':0,'Bearing2_2':0,
                'Bearing3_1':0,'Bearing3_2':0,
                'Bearing1_3':573,'Bearing1_4':33.9,'Bearing1_5':161,'Bearing1_6':146,'Bearing1_7':757,
                'Bearing2_3':753,'Bearing2_4':139,'Bearing2_5':309,'Bearing2_6':129,'Bearing2_7':58,
                'Bearing3_3':82}
    phm_dataset = DataSet(name='phm_data',
                        index=['bearing_name','RUL','quantity','data'])
    source_path = './PHM/'
    for path_1 in ['Learning_set/','Test_set/']:
        bearings_names = os.listdir(source_path + path_1)
        bearings_names.sort()
        for bearings_name in bearings_names:
            file_names = os.listdir(source_path + path_1 + bearings_name + '/')
            file_names.sort()
            bearing_data = np.array([])
            for file_name in file_names:
                if 'acc' in file_name:
                    df = pd.read_csv(source_path + path_1 + bearings_name + '/'\
                                    + file_name,header=None)
                    data = np.array(df.loc[:,4:6])
                    data = data[np.newaxis,:,:]
                    if bearing_data.size == 0:
                        bearing_data = data
                    else:
                        bearing_data = np.append(bearing_data,data,axis=0)
        
            phm_dataset.append([bearings_name,RUL_dict[bearings_name],bearing_data.shape[0],bearing_data])
            print(bearings_name,'has been appended.')

    phm_dataset.save()

if __name__ == '__main__':
    # make_phm_dataset()
    dataset = DataSet.load_dataset('phm_data')
    print('1')