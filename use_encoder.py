# -*- coding: utf-8 -*-
"""
Created on Sun Apr 22 22:36:18 2018

@author: Rishabh
"""
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import model_from_json
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import h5py
import csv
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

autoencoder = model_from_json(json_str)

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

#autoencoder.save("saved_models/encoder.h5")
#print(" Auto encoder saved succesfully ")


#Get the encoded data, to be fed into a NN
X_encoded = autoencoder.predict(X_original)

#print(X_encoded.shape)
#Y_original = Y_original.values.reshape(22566,1)
#print(Y_original.shape)
#encoded_data = np.concatenate((X_encoded,Y_original),axis=1)
#np.savetxt("data/encoded.csv",encoded_data,delimiter=",")


X_encoded = X_encoded.tolist()


Y_encoded=Y_original.tolist()
#print([Y_encoded[87:95]])

encoded_data=[]
for x,y in zip(X_encoded,Y_original):
    encoded_data.append(X_encoded[87] + [Y_original[87]])

print(encoded_data[89:91])


encoded_data=np.array(encoded_data)

X_train,X_test = train_test_split(encoded_data,test_size=0.05)

Y_train=X_train[:,-1]
X_train=np.delete(X_train,-1,axis=1)

Y_test=X_test[:,-1]
X_test=np.delete(X_test,-1,axis=1)
#print(X_train[1:6])

print(X_train.shape,X_test.shape)

dim_input = X_train.shape[1]

#Y_train=to_categorical(Y_train,num_classes=2)
#Y_test=to_categorical(Y_test,num_classes=2)


##Define our new NN

#Define the parameters first
epochs = 75
batch_size=32

classifier = Sequential()

classifier.add(Dense(5,input_shape=(dim_input,),activation="tanh"))

#classifier.add(Dense(5,activation="tanh"))

classifier.add(Dense(1,activation="sigmoid"))

classifier.summary()

model_json = classifier.to_json()
#with open("saved_models/classifier.json", "w") as json_file:
 #   json_file.write(model_json)

classifier.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

trained = classifier.fit(X_train,Y_train,epochs=epochs,batch_size=batch_size,verbose=1,validation_data=(X_test,Y_test))

#classifier.save_weights("saved_models/classifierweights.h5")
print("Saved model and weights to disk")

test_eval = classifier.evaluate(X_test,Y_test)
print('Test loss : ', test_eval[0])
print('test accuracy: ', test_eval[1])

print(X_test[1])
print(classifier.predict(X_test[1]))

