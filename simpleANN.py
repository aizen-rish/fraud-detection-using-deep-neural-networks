# -*- coding: utf-8 -*-
"""
Created on Sat Apr 28 11:36:37 2018

@author: Rishabh
"""

from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.models import model_from_json
from keras.models import load_model
from keras.layers import Dense
from keras.layers import LeakyReLU
from keras.utils import to_categorical
import numpy as np
import pandas as pd
import h5py


# Load model
handler = open("saved_models/classifier.json","r")
json_str =  handler.read()

fmodel = model_from_json(json_str)

print("Loaded classifier succesfully")

fmodel.load_weights("saved_models/classifierweights.h5")

dt=pd.read_csv("data/encoded.csv",header=None)


print(dt[1:2])

xnow=dt[1:2].drop(dt.columns[-1],axis=1)
answer=fmodel.predict(xnow)
answer[answer < 0.5]=0
print(answer)

'''
X_test = dt.drop(dt.columns[-1],axis=1)
Y_test = dt[dt.columns[-1]]

print(" Running the ANN on the whole dataset ")
fmodel.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])
answer = fmodel.evaluate(X_test,Y_test)

print('test accuracy: ', answer[1]*100, '%')
'''
