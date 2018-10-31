import numpy as np
import random
import matplotlib.pyplot as plt
import keras
from dataset import DataSet
import keras.layers as KL
from keras import backend as K
import tensorflow as tf 
import keras.backend.tensorflow_backend as KTF

class CNN_GRU():
    def __init__(self):
        self.input_shape = (2560,2)
        self.feature_size = 16
        self.dataset = DataSet.load_dataset(name='phm_data')
        self.train_bearings = ['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']
        self.test_bearings = ['Bearing1_3','Bearing1_4','Bearing1_5','Bearing1_6','Bearing1_7',
                                'Bearing2_3','Bearing2_4','Bearing2_5','Bearing2_6','Bearing2_7',
                                'Bearing3_3']
    
    def _build_cnn(self):
        inp = KL.Input(shape=(self.input_shape))
        x = inp
        x = KL.Conv1D(32,16,activation='relu')(x)
        x = KL.MaxPool1D(4)(x)
        x = KL.Conv1D(64,3,activation='relu')(x)
        x = KL.MaxPool1D(4)(x)
        x = KL.Conv1D(64,3,activation='relu')(x)
        x = KL.MaxPool1D(2)(x)
        x = KL.Conv1D(64,3,activation='relu')(x)
        x = KL.MaxPool1D(2)(x)
        x = KL.Conv1D(64,3,activation='relu')(x)
        x = KL.MaxPool1D(2)(x)
        x = KL.Flatten()(x)
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(128,activation='relu')(x)
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(self.feature_size,activation='relu',name='feature')(x)
        out = KL.Dense(1)(x)
        
        model = keras.Model(inp,out)
        model.summary()
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                        loss='mse')
        return model

    def _build_gru(self):
        inp = KL.Input(shape=(100,self.feature_size))
        x = inp
        x = KL.Masking()(x)
        x = KL.GRU(32,return_sequences=True)(x)
        x = KL.GRU(1)(x)
        out = x

        model = keras.Model(inp,out)
        model.compile(optimizer=keras.optimizers.Adam(lr=0.001),
                        loss='mse')
        return model

    def _c_preprocess(self,select='train',is_random=True):
        if select == 'train':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.train_bearings})
        elif select == 'test':
            temp_data = self.dataset.get_value('data',condition={'bearing_name':self.test_bearings})
            temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.test_bearings})
        else:
            raise ValueError('wrong selection!')
        # temp_data = self.dataset.get_value('data',condition={'bearing_name':self.train_bearings})
        # temp_label = self.dataset.get_value('RUL',condition={'bearing_name':self.train_bearings})
        train_data = np.array([])
        train_label = np.array([])
        for i,x in enumerate(temp_label):
            t_label = [y for y in range(x,x + temp_data[i].shape[0])]
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
        return train_data,train_label
    
    def _g_preprocess(self,select):
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
        cnn_feature = keras.Model(self.cnn.input,self.cnn.get_layer('feature').output)
        for i,x in enumerate(temp_label):
            t_label = [y for y in range(x,x + temp_data[i].shape[0])]
            t_label.reverse()
            r_temp_label.append(np.array(t_label))
            r_temp_data.append(cnn_feature.predict(temp_data[i]))

        r_data = []
        r_label = []
        for i in range(10000):
            bearing_idx = random.randint(0,len(r_temp_data)-1)
            random_bearing = r_temp_data[bearing_idx]
            random_bearing_RUL = r_temp_label[bearing_idx]
            start_idx = random.randint(0,random_bearing.shape[0]-101)
            end_idx = start_idx + random.randint(50,100)
            r_t_data = random_bearing[start_idx:end_idx,]
            if r_t_data.shape[0] < 100:
                r_t_data = np.append(np.zeros((100-r_t_data.shape[0],self.feature_size)),r_t_data,axis=0)
            r_data.append(r_t_data)
            r_label.append(random_bearing_RUL[end_idx])

        return np.array(r_data),np.array(r_label)
        

    def train(self):
        c_train_data,c_train_label = self._c_preprocess()
        self.cnn = self._build_cnn()
        self.cnn.fit(c_train_data,c_train_label,batch_size=32,epochs=50)

        g_train_data,g_train_label = self._g_preprocess('train')
        self.gru = self._build_gru()
        self.gru.fit(g_train_data,g_train_label,batch_size=32,epochs=50)

    def test(self):
        test_data,test_label = self._g_preprocess('test')
        self.gru.evaluate(test_data,test_label)
    
    def save(self):
        self.cnn.save_weights('./weights/cnn.h5')
        self.gru.save_weights('./weights/gru.h5')

    def test_cnn(self):
        c_train_data,c_train_label = self._c_preprocess()
        self.cnn = self._build_cnn()
        self.cnn.fit(c_train_data,c_train_label,batch_size=32,epochs=50)
    
        c_test_data,c_test_label = self._c_preprocess('test',False)
        predict_label = self.cnn.predict(c_test_data)

        plt.plot(c_test_label)
        plt.scatter([x for x in range(predict_label.shape[0])],predict_label)

if __name__ == '__main__':
    KTF.set_session(tf.Session(config=tf.ConfigProto(device_count={'gpu':0})))
    process = CNN_GRU()
    process.test_cnn()