# -*- coding: utf-8 -*-
"""
Created on Thu May 14 03:02:49 2020

@author:Abhishek Mukherjee
"""

import pandas as pd
df=pd.read_csv('C:/Users/abhi0/OneDrive/Documents/Tweets_dataset/train.csv')

import numpy as np

y= np.array(df['target'])

import matplotlib.pyplot as plt
plt.hist(y)
plt.show()

import numpy as np
from sklearn.model_selection import train_test_split

# The maximum number of words to be used. (most frequent)
from keras.preprocessing.text import Tokenizer
MAX_NB_WORDS = 50000

# Max number of words in each headline.
MAX_SEQUENCE_LENGTH = 300

# This is fixed.
EMBEDDING_DIM = 300

tokenizer =\
Tokenizer(num_words=MAX_NB_WORDS, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True)
tokenizer.fit_on_texts(df['text'])
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

from keras.preprocessing.sequence import pad_sequences
X = tokenizer.texts_to_sequences(df['text'].values)
X = pad_sequences(X, maxlen=MAX_SEQUENCE_LENGTH)
print('Shape of data tensor:', X.shape)
y= np.array(df['target'])

from keras.utils import to_categorical
from sklearn import preprocessing
le = preprocessing.LabelEncoder()
y=le.fit_transform(y)
print(y.shape)

x_train, x_test, y_train, y_test =\
train_test_split(X,y, test_size = 0.10, random_state = 42,shuffle=True)

from keras.layers import SpatialDropout1D
from keras.layers.recurrent import LSTM
from keras.callbacks import EarlyStopping
from keras.callbacks import LearningRateScheduler
from keras.callbacks import BaseLogger
import matplotlib.pyplot as plt
from keras.callbacks import ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.callbacks import ModelCheckpoint
from keras.layers import TimeDistributed
from keras.regularizers import L1L2

model = Sequential()
model.add(Embedding(MAX_NB_WORDS, EMBEDDING_DIM, input_length=X.shape[1]))
print(X.shape[1])
model.add(SpatialDropout1D(0.80))
model.add(LSTM(200,dropout=0.80,recurrent_dropout=0.80,return_sequences=True,\
               recurrent_regularizer=L1L2(l1=0.8, l2=0.8)))
model.add(SpatialDropout1D(0.80))
model.add(LSTM(200,dropout=0.80,return_sequences=True,recurrent_dropout=0.80,\
               recurrent_regularizer=L1L2(l1=0.8, l2=0.8)))
model.add(SpatialDropout1D(0.80))
#
from keras.layers import Flatten
model.add(TimeDistributed(Dense(200)))
model.add(Flatten())
model.add(Dense(1,activation='sigmoid'))


model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

mdl_chk=\
ModelCheckpoint('C:/Users/abhi0/OneDrive/Documents/nlp_disaster_tweets_real_fake/BestModel_RNN_real_fake.h5',\
                monitor='val_accuracy',\
                verbose=1,\
                save_best_only=True,\
                save_weights_only=True)

history =\
model.fit(x_train, y_train, epochs=1000, batch_size=10,\
          validation_split=0.4,\
          callbacks= [mdl_chk])

plt.title('Accuracy')
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()    

