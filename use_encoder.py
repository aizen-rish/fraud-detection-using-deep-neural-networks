# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 22:36:18 2018

@author: Rishabh
"""
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LeakyReLU
import keras.utils
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

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
#print(len(dt[dt['fraud']==1]))

Y_original = dt['fraud']
X_original = dt.drop(['fraud'],axis=1)

#print(Y_original[89:91])


# Load model
handler = open("saved_models/model_with_frauds.json","r")
json_str =  handler.read()

autoencoder = keras.models.model_from_json(json_str)

print("Loaded succesfully")

#Load weights
autoencoder.load_weights("saved_models/mweights_with_frauds.h5")

#Remove the decoder part to get only encoder part
autoencoder.layers.pop()
autoencoder.layers.pop()
autoencoder.layers.pop()
autoencoder.layers.pop()

autoencoder.outputs= [autoencoder.layers[-1].output]
autoencoder.layers[-1].outbound_nodes = []


#Get the encoded data, to be fed into a NN
X_encoded = autoencoder.predict(X_original)
X_encoded = X_encoded.tolist()
#print(X_encoded[1])
Y_original=Y_original.tolist()
#print(Y_original[87:91])


#encoded_data = 
for x,y in zip(X_encoded,Y_original):
    ass=x.append(y)
    print(ass)

#print(encoded_data[1])
#X_train,X_test = train_test_split(encoded_data,test_size=0.05)
#
#print(X_train.shape,X_test.shape)
#
#dim_input = X_train.shape[1]



##Define our new NN
#classifier = Sequential()
#
#classifier.add(Dense(7,input_shape=(dim_input,),activation="tanh"))
#
#classifier.add(Dense(5,activation="tanh"))
#
#classifier.add(Dense(2,activation="sigmoid"))
#
#classifier.summary()
