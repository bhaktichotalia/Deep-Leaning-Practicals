# -*- coding: utf-8 -*-
""" Created on 12/02/2022 by Bhakti Chotalia """

import pandas
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

# load dataset
dataframe = pandas.read_csv("Iris.csv", header=None, skiprows=1)
dataset = dataframe.values
X = dataset[:,1:5].astype(float)
Y = dataset[:,5]
print("\nX : ",X)
print("\nY : ",Y)

# encode class values as integers
encoder = LabelEncoder()
encoder.fit(Y)
encoded_Y = encoder.transform(Y)

# convert integers to dummy variables (i.e. one hot encoded)
dummy_y = np_utils.to_categorical(encoded_Y)
Input_train, Input_test, Target_train, Target_test = train_test_split(X, dummy_y, test_size = 0.30, random_state = 5)

# define baseline model
# create model
model = Sequential()
model.add(Dense(8, input_dim=4, activation='relu'))
model.add(Dense(4 ,activation='relu'))
model.add(Dense(3, activation='sigmoid'))

# Compile model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(Input_train, Target_train, epochs=200, verbose=1)
model.summary()
score = model.evaluate(Input_test, Target_test, verbose=0)
print('Model Accuracy = ',score[1]*100)
