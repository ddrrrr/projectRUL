import random
from dataset import DataSet
import numpy as np
from collections import deque

class RUL_Predict():
    def __init__(self,data_name):
        self.dataset = DataSet.load_dataset(name=data_name)

    def reset(self,stage):
        assert stage < 8
        self.pred_RUL = min(500*stage,3000)
        self.statement = deque([[0.0]]*2000,maxlen=2000)
        self.chosen_data = self.dataset.get_random_choice()
        self.RUL = self.chosen_data['RUL']
        self.index = max(self.chosen_data['data'].shape[0]-1-random.randint(500*(stage-1),self.pred_RUL),0)
        print('the chosen bearing is :',self.chosen_data['bearing_name'])
        self.statement.append([self.pred_RUL])
        return [self.chosen_data['data'][self.index,:,:],np.array(self.statement)]
    
    def step(self,action):
        self.pred_RUL = self.pred_RUL * (1 + (action-10) / 100) - 1
        self.statement.append([self.pred_RUL])
        if self.index == self.chosen_data['data'].shape[0] - 1:
            done = True
            self.real_RUL = self.RUL
            reward = -(self.pred_RUL - self.real_RUL)**2
            _s = np.zeros(np.shape(self.chosen_data['data']))
        elif self.pred_RUL < 0:
            done = True
            self.real_RUL = self.chosen_data['data'].shape[0] - self.index + self.RUL
            reward = -(self.pred_RUL - self.real_RUL)**2
            _s = np.zeros(np.shape(self.chosen_data['data']))
        elif self.chosen_data['data'].shape[0] - 1 - self.index < 50:
            done = False
            self.real_RUL = self.chosen_data['data'].shape[0] - self.index + self.RUL
            reward = -(self.pred_RUL - self.real_RUL)**2
            _s = self.chosen_data['data'][self.index,:,:]
        else:
            done = False
            reward = 0
            self.index = self.index + 1
            _s = self.chosen_data['data'][self.index,:,:]
        return done,reward/100,[_s,np.array(self.statement)]