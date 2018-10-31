import random
import numpy as np
from env import RUL_Predict
from dataset import DataSet
from collections import deque
import keras.layers as KL
from keras.models import Model
from keras.optimizers import Adam
from keras import backend as K

import tensorflow as tf

EPISODES = 5000

class DQNAgent:
    def __init__(self, state_size, action_size,statement_size):
        self.state_size = state_size
        self.action_size = action_size
        self.statement_size = statement_size
        self.memory = deque(maxlen=20000)
        self.gamma = 0.99    # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.05
        self.epsilon_decay = 0.99
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.target_model = self._build_model()
        self.update_target_model()

    """Huber loss for Q Learning
    References: https://en.wikipedia.org/wiki/Huber_loss
                https://www.tensorflow.org/api_docs/python/tf/losses/huber_loss
    """

    def _huber_loss(self, y_true, y_pred, clip_delta=1.0):
        error = y_true - y_pred
        cond  = K.abs(error) <= clip_delta

        squared_loss = 0.5 * K.square(error)
        quadratic_loss = 0.5 * K.square(clip_delta) + clip_delta * (K.abs(error) - clip_delta)

        return K.mean(tf.where(cond, squared_loss, quadratic_loss))

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        inp = KL.Input(shape=(self.state_size))
        x = inp
        x = KL.Conv1D(64,72,strides=8,activation='relu')(x)
        x = KL.Conv1D(64,12,strides=4,activation='relu')(x)
        x = KL.Conv1D(128,7,strides=3,activation='relu')(x)
        x = KL.Conv1D(128,3,strides=3,activation='relu')(x)
        x = KL.Conv1D(256,3,activation='relu')(x)
        x = KL.MaxPool1D(3)(x)
        x = KL.Flatten()(x)
        x = KL.Dropout(0.3)(x)
        x = KL.Dense(64,activation='relu')(x)

        inp_R = KL.Input(shape=(self.statement_size,1))
        R = inp_R
        R = KL.Masking()(R)
        R = KL.GRU(64)(R)

        out = KL.Add()([x,R])
        out = KL.Dense(128,activation='relu')(out)
        out = KL.Dense(self.action_size)(out)

        model = Model([inp,inp_R],out)

        model.compile(loss=self._huber_loss,
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def update_target_model(self):
        # copy weights from model to target_model
        self.target_model.set_weights(self.model.get_weights())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randint(0,self.action_size-1)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(state)
            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)[0]
                target[0][action] = reward + self.gamma * np.amax(t)
            self.model.fit(state, target, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == "__main__":
    env = RUL_Predict('phm_data')
    env.dataset.dataset_filter({'bearing_name':['Bearing1_1','Bearing1_2','Bearing2_1','Bearing2_2','Bearing3_1','Bearing3_2']})
    env.dataset.normalization('data')
    state_size = (2560,2)
    action_size = 11
    statement_size = 2000
    agent = DQNAgent(state_size, action_size,statement_size)
    done = False
    batch_size = 32

    for e in range(EPISODES):
        state = env.reset(min(7,e//300+1))
        for i,x in enumerate(state):
            state[i] = np.reshape(x,(1,)+np.shape(x))
        # state = state[np.newaxis,:,:]
        done = False
        i = 0
        while True:
        # for time in range(500):
            i = i+1
            action = agent.act(state)
            done, reward, next_state = env.step(action)
            for j,x in enumerate(next_state):
                next_state[j] = np.reshape(x,(1,)+np.shape(x))
            # next_state = next_state[np.newaxis,:,:]
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            if done:
                agent.update_target_model()
                print("episode: {}/{}, RUL: {}, pred_RUL: {:.5}, reward: {:.5}, e: {:.2}, i: {}"
                      .format(e, EPISODES, env.real_RUL, env.pred_RUL, reward, agent.epsilon, i))
                break
            if len(agent.memory) > batch_size*50 and i%3==0:
                agent.replay(batch_size)
