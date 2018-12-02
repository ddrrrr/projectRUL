# -*- coding: utf-8 -*-
"""
Created on Mon Aug  6 14:53 2018

@author: a273
TODO
    should class DataSet only arange, save and load?
"""

import os
import operator
import random
import scipy.io as sio
import pickle as pickle
import numpy as np
import pandas as pd
from collections import OrderedDict

class DataSet(object):
    '''This class is used to arrange dataset, collected and used by Lab 119 in HIT.
        module numpy, pickle and pandas should be installed before used.
        Attributes:
            name: The name of dataset with str type. And this name is used to save and load the 
                dataset with the file name as 'DataSet_' + name + '.pkl'
            index: A list contained atrributes of the dataset, so that samples can be distinguished 
                from each others by different values under same attributes.
            save_path: A string described where to save or load this dataset, and defaulted as './data/'
            dataset: A list contained samples and their attributes.
    '''
    def __init__(self,name='',index=[],save_path='./data/',dataset=[]):
        self.name = name
        self.index = index
        self.save_path = save_path
        self.dataset = dataset

    # inner function
    def _deal_condition(self,condition):
        '''
        get the index of samples whose attributes is in condition.

        Args:
            condition: A dict whose keys are the name of attributes and values are lists contained values owned by 
                samples we need.
        Return:
            A bool list whether the sample need according to condition.
        '''
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
        '''
        Add new attribute to dataset.
        
        Args:
            new_attribute: The name of new attribute (a string).
            new_value: A list contained values appended to each sample. If the length of new_value is 1,
                then all samples will append the same new_value. Or the length of new_value should be the
                same as the number of samples, then each sample will append the corresponding value. 
                Otherwise, raise valueError.
        Return:
            None
        '''
        self.index.append(new_attribute)
        if new_value == None:
            for x in self.dataset:
                x.append(new_value)
        elif isinstance(new_value,list):
            if len(new_value) == 1:
                for i in range(len(self.dataset)):
                    self.dataset[i].append(new_value[0])
            elif len(new_value) == len(self.dataset):
                for i in range(len(self.dataset)):
                    self.dataset[i].append(new_value[i])
            else:
                raise TypeError

    def del_index(self,del_attribute):
        '''
        delete attribute and the corresponding values in each sample.
        
        Args:
            del_attribute: The name of attribute (a string).
        Return:
            None
        '''
        try:
            idx = self.index.index(del_attribute)
            for x in self.dataset:
                del(x[idx])
            self.index.remove(del_attribute)
        except ValueError:
            raise ValueError
            print('The given attribute does not exist in index, and the attributes of this dataset \
                is ', self.index)

    def append(self,append_data):
        '''
        Append samples.
        
        Args:
            append_data: A dict or a list that contain a sample, including data and attribute.
        Return:
            None
        '''
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
        '''
        delete samples.
        
        Args:
            condition: A dict determines which samples should be delete.
        Return:
            None
        '''
        conforming_idx = self._deal_condition(condition)
        for i,x in enumerate(self.dataset):
            if conforming_idx[i]:
                self.dataset.pop(x)

    # get information or values
    def get_value_attribute(self,attribute):
        '''
        get values under the given attribute of each data
        Args:
            attribute: A str mapping the attribute of dataset.
                Return error and all attribute of dataset if the given attribute does
                not exist.
        Return:
            A list of values under the given attribute with the same order as samples in dataset.
        '''
        try:
            idx = self.index.index(attribute)
            return [x[idx] for x in self.dataset]
        except ValueError:
            raise ValueError
            print('The given attribute does not exist in index, and the attributes of this dataset \
                is ', self.index)

    def get_value(self,attribute,condition={}):
        '''
        get corresponding values.
        
        Args:
            attribute: A string describes the values returned.
            condition: A dict determines the values of which samples should be returned.
        Return:
            A list contrained values by given attribute and condition.
        '''
        conforming_idx = self._deal_condition(condition)
        idx = self.index.index(attribute)
        return [x[idx] for i,x in enumerate(self.dataset) if conforming_idx[i]]

    def get_dataset(self,condition={}):
        '''
        get corresponding dataset.
        
        Args:
            condition: A dict determines the values of which samples should be returned.
        Return:
            A DataSet contrained values by given condition.
        '''
        conforming_idx = self._deal_condition(condition)
        return DataSet(name='temp',index=self.index,
                        dataset=[x for i,x in enumerate(self.dataset) if conforming_idx[i]])

    def get_random_choice(self):
        '''
        get a random sample.
        
        Args:
            None
        Return:
            A dict like {Attribute_1:Values,...}.
        '''
        r = {}
        data = random.choice(self.dataset)
        for i,k in enumerate(self.index):
            r[k] = data[i]
        return r

    def get_random_samples(self,n=1):
        '''
        get a random DataSet.
        
        Args:
            None
        Return:
            A Dataset with same index but only one sample.
        '''
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
        '''
        Save this DataSet as .pkl file.
        
        Args:
            None
        Return:
            None
        '''
        assert self.name != ''
        assert self.save_path != ''
        pickle.dump(self, open(self.save_path + 'DataSet_' +
                                     self.name + '.pkl', 'wb'), True)
        self._save_info()
        print('dataset ', self.name, ' has benn saved\n')

    def _save_info(self):
        '''
        Save this DataSet' information as .csv file in the save_path.
        
        Args:
            None
        Return:
            None
        '''
        assert self.name != ''
        assert self.save_path != ''
        info = OrderedDict()
        for attr in self.index:
            info[attr] = self.get_value_attribute(attr)
            if isinstance(info[attr][0],np.ndarray) and len(info[attr][0])>1:
                for i,x in enumerate(info[attr]):
                    info[attr][i] = x.shape
            if not isinstance(info[attr][0],str) and len(info[attr][0]) > 2:
                for i,x in enumerate(info[attr]):
                    info[attr][i] = len(x)

        pd.DataFrame(info).to_csv(self.save_path + 'DataSet_' + self.name + 'info.csv',index=False)

    def load(self,name=''):
        '''
        Load this DataSet with name and path known, which should be given when initialize DataSet class.
        
        Args:
            name: The name of DataSet.
        Return:
            None
        '''
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
        '''
        Load this DataSet with name and default path './data/'.
        
        Args:
            name: The name of DataSet.
        Return:
            DataSet
        '''
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

def make_paderborn_dataset():
    paderborn_dataset = DataSet(
        name='paderborn_data',
        index=[
            'bearing_name',
            'load',
            'speed',
            'fault_place',
            'fault_cause',
            'state',
            'No',
            'data'
        ]
    )
    source_path = 'E:/cyh/data_sum/temp/德data/dataset/'
    artificial_fault = ['KI01','KI03','KI05','KI07','KI08',
                'KA01','KA03','KA05','KA06','KA07']
    state = {
        'K001':'>50',
        'K002':19,
        'K003':1,
        'K004':5,
        'K005':10,
        'K006':16,
        'KA01':'EMD',
        'KA03':'Electric Engraver',
        'KA05':'Electric Engraver',
        'KA06':'Electric Engraver',
        'KA07':'Drilling',
        'KA08':'Drilling',
        'KA09':'Drilling',
        'KA04':'Pitting',
        'KA15':'Plastic Deform',
        'KA16':'Pitting',
        'KA22':'Pitting',
        'KA30':'Plastic Deform',
        'KI01':'EMD',
        'KI03':'Electric Engraver',
        'KI05':'Electric Engraver',
        'KI07':'Electric Engraver',
        'KI08':'Electric Engraver',
        'KI04':['Pitting','Plastic Deform'],
        'KI14':['Pitting','Plastic Deform'],
        'KI16':'Pitting',
        'KI17':'Pitting',
        'KI18':'Pitting',
        'KI21':'Pitting',
        'KB23':'Pitting',
        'KB24':'Pitting',
        'KB27':'Plastic Deform'
    }
    file_names = os.listdir(source_path)
    file_names.sort()
    for file_name in file_names:
        temp_data = sio.loadmat(source_path + file_name)
        temp_data = temp_data[file_name.replace('.mat','')]
        temp_data = temp_data['Y']
        temp_data = temp_data[0][0][0][6][2][0]
        temp_fault_cause = 'artificial' if file_name[12:16] in artificial_fault \
                                        else 'real'
        temp_append_sample = [
            file_name,
            file_name[4:11],
            file_name[0:3],
            file_name[12:16],
            temp_fault_cause,
            state[file_name[12:16]],
            file_name[17:],
            temp_data
        ]
        paderborn_dataset.append(temp_append_sample)
        print(file_name,'has been appended.')

    paderborn_dataset.save()

def make_ims_dataset():
    fault_bearing = {'1st_test':OrderedDict({4:'3_x',5:'3_y',6:'4_x',7:'4_y'}), '2nd_test':[0], '4th_test':[2]}
    ims_dataset = DataSet(name='ims_data', index=['set_No','bearing_No','record_time','data'])
    source_path = 'E:/cyh/data_sum/temp/IMS data/'

    for dir_name in fault_bearing.keys():
        # if isinstance(fault_bearing[dir_name],dict):
        #     all_samples = []
        #     for k in fault_bearing[dir_name].keys():
        #         all_samples.append([dir_name, fault_bearing[dir_name][k], [], []])
        if isinstance(fault_bearing[dir_name], dict):
            channels = list(fault_bearing[dir_name].keys())
        elif isinstance(fault_bearing[dir_name], list):
            channels = fault_bearing[dir_name]
        record_time = []
        record_data = np.array([])

        names = os.listdir(source_path + dir_name + '/')
        names.sort()
        for name in names:
            print(name)
            record_time.append(name.replace('.txt',''))
            temp_data = np.loadtxt(source_path + dir_name + '/' + name)
            if record_data.size == 0:
                record_data = temp_data[:,channels][np.newaxis,:,:]
            else:
                record_data = np.append(record_data, temp_data[:,channels][np.newaxis,:,:], axis=0)
        
        if isinstance(fault_bearing[dir_name], dict):
            append_samples = []
            for i,k in enumerate(fault_bearing[dir_name].keys()):
                append_samples.append([dir_name, fault_bearing[dir_name][k], record_time, record_data[:,:,i]])
        elif isinstance(fault_bearing[dir_name], list):
            append_samples = []
            for i,x in enumerate(fault_bearing[dir_name]):
                append_samples.append([dir_name, str(x), record_time, record_data[:,:,i]])

        for sample in append_samples:
            ims_dataset.append(sample)

    ims_dataset.save()

            


if __name__ == '__main__':
    # make_phm_dataset()
    # dataset = DataSet.load_dataset('phm_data')
    # dataset._save_info()
    # make_paderborn_dataset()
    # make_ims_dataset()
    dataset = DataSet.load_dataset('ims_data')
    print('1')