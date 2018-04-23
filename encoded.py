# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 23:12:13 2018

@author: Rishabh
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Input,Dense
from keras.layers import LeakyReLU
from keras.callbacks import ModelCheckpoint
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

import pandas as pd

Labels= ['normal','fraud']

dt=pd.read_csv("data/data_small.csv")

#print(dt.shape)
#Output (22566,10)  10 columns

count_classes = pd.value_counts(dt['fraud'],sort=True)
print(count_classes)

frauds = dt[dt.fraud == 1]
normal = dt[dt.fraud == 0]

#plt.bar(np.arange(2),count_classes)
#plt.xticks(range(2),Labels)
#plt.ylabel('count')
#plt.xlabel('Class')
#plt.show()

#Lets drop zipcodeOri and zipMerchant
dt = dt.drop(['zipcodeOri'],axis=1)
dt = dt.drop(['zipMerchant'],axis=1)

#Scale amount column
st=StandardScaler()
dt['amount'] = st.fit_transform(dt['amount'].values.reshape(-1,1))

labelencoder = LabelEncoder()
dt['merchant']=labelencoder.fit_transform(dt['merchant'])
dt['customer']=labelencoder.fit_transform(dt['customer'])
print(dt[:10])

#Now make train and test samples (15%)
x_train, x_test = train_test_split(dt,test_size=0.05)

#Train only on normal transactions, without classes
x_train = x_train[x_train.fraud==0]
x_train = x_train.drop(['fraud'],axis=1)

y_test = x_test['fraud']
x_test = x_test.drop(['fraud'],axis=1)
print(x_train.shape,x_test.shape)

batch_size = 64
epochs = 100
dim = x_train.shape[1]


autoencoder = Sequential()

autoencoder.add(Dense(10,input_shape=(dim,),activation='tanh'))

autoencoder.add(Dense(14,activation="linear"))
autoencoder.add(LeakyReLU(alpha=0.1))

autoencoder.add(Dense(9,activation="linear"))
autoencoder.add(LeakyReLU(alpha=0.1))

autoencoder.add(Dense(4,activation="tanh"))

autoencoder.add(Dense(9,activation="tanh"))

autoencoder.add(Dense(14,activation="tanh"))

autoencoder.add(Dense(dim,activation="linear"))
autoencoder.add(LeakyReLU(alpha=0.1))

autoencoder.summary()

model_json = autoencoder.to_json()
with open("saved_models/model.json", "w") as json_file:
    json_file.write(model_json)

autoencoder.compile(loss='mean_squared_logarithmic_error',optimizer='Adam',metrics=['accuracy'])

checkpointer = ModelCheckpoint(filepath="model.h5",verbose=0,save_best_only=True)

auto_train = autoencoder.fit(x_train,x_train,epochs=epochs,batch_size=batch_size,verbose=1,validation_data=(x_test,x_test))

autoencoder.save_weights("saved_models/model.h5")
print("Saved model to disk")

test_eval = autoencoder.evaluate(x_test,x_test)
print('Test loss : ', test_eval[0])
print('test accuracy: ', test_eval[1])
