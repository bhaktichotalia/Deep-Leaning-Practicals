# -*- coding: utf-8 -*-
""" Created on 23/02/2022 by Bhakti Chotalia """

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
dataset = pd.read_csv('diabetes.csv')

# creating input features and target variables
X= dataset.iloc[:,0:8]
y= dataset.iloc[:,8]

#standardizing the input feature
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X = sc.fit_transform(X)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
from keras import Sequential
from keras.layers import Dense
from keras.regularizers import l2
classifier = Sequential()

#First Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal',input_dim=8, kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))

#Second Hidden Layer
classifier.add(Dense(4, activation='relu', kernel_initializer='random_normal',kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))

#Output Layer
classifier.add(Dense(1, activation='sigmoid', kernel_initializer='random_normal',kernel_regularizer=l2(0.001), bias_regularizer=l2(0.001)))

#Compiling the neural network
classifier.compile(optimizer ='adam',loss='binary_crossentropy', metrics=['accuracy'])

#Fitting the data to the training dataset
history = classifier.fit(X_train,y_train, validation_data=(X_test, y_test),batch_size=10, epochs=100)
_, train_acc = classifier.evaluate(X_train, y_train, verbose=0)
_, test_acc = classifier.evaluate(X_test, y_test, verbose=0)
print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='test')
plt.legend()
plt.show()
