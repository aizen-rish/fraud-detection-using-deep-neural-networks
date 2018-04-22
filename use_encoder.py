# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 22:36:18 2018

@author: Rishabh
"""
from sklearn.model_selection import train_test_split
from keras.models import Sequential
import keras.utils
import pandas as pd

#Test on some data
dt=pd.read_csv("data/data_small.csv")
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

# Load model
handler = open("model.json","r")
json_str =  handler.read()

autoencoder = keras.models.model_from_json(json_str)

print("Loaded succesfully")

#Load weights
autoencoder.load_weights("model.h5")

print(x_test[:1])
#print(autoencoder.predict(x_test[:1]))

#Remove the decoder part to get only encoder part
autoencoder.layers.pop()
autoencoder.layers.pop()
autoencoder.layers.pop()
autoencoder.layers.pop()

autoencoder.outputs= [autoencoder.layers[-1].output]
autoencoder.layers[-1].outbound_nodes = []

print(autoencoder.predict(x_test[:1]))
