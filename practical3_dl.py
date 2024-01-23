# -*- coding: utf-8 -*-
""" Created on 03/02/2022 by Bhakti Chotalia """


import pandas as pd
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix
from sklearn.compose import ColumnTransformer
import numpy as np

# Import the data 
dataset = pd.read_csv("Churn_Modelling.csv")
print(dataset.head())

# Feature selection
X = dataset.iloc[:, 3: 13].values   # Independent variable X 
y = dataset.iloc[:, 13].values   # Dependent variable y

# Categorical variables, such as Geography and Gender need to be encoded into numerical ones. 
# Here we use fit_transform() method of LabelEncoder from sklearn.
labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])
labelencoder_X_2 = LabelEncoder()

# Notice above, we input index 1, as the index of Geography column in X is 1. 
# After encoding, country German becomes 1, France is 0, Spain is 2. With the same method, encode Gender column
X[:, 2] = labelencoder_X_2.fit_transform(X[:, 2])

# ColumnTransformer() implements the transform function and takes as input the column name, the transformer (OneHotEncoder in this case), 
# and the number of columns to be transformed this way; i.e. with unique combinations of 0s and 1s.
ct = ColumnTransformer([('one_hot_encoder', OneHotEncoder(categories='auto'), [1])], remainder='passthrough')                                         
X = ct.fit_transform(X)
X = X[:, 1:]

# split the data into training and test set with the test set taking 20%. We use random_state to make sure splitting remains the same each time.
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size = 0.2, random_state =0)

# Scaling the features is to avoid intensive computation and also avoid one variable dominating the others. 
# Here we take Standardization.
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Initialize the sequential model.
classifier = Sequential()

# The model is built with 2 dense layers.
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu', input_dim= 11))   # Add input layer and first hidden layer
classifier.add(Dense(6, kernel_initializer = 'uniform', activation = 'relu'))     # Add 2nd hidden layer
classifier.add(Dense(1, kernel_initializer = 'uniform', activation = 'sigmoid'))   # Add output layer

# Model compiling is to apply Stochastic Gradient Descent (SGD) on the network.
# Here we use Adam (one type of SGD) as an optimizer to find the optimized weights that make neural network most powerful. 
# The loss function that the optimizer is based on is binary cross-entropy. 
# The metrics we use to evaluate the model performance are accuracy.
classifier.compile(optimizer = 'Adam', loss ='binary_crossentropy', metrics =['accuracy'])

# Since we use SGD, the batch size is set to 10, indicating neural network updates its weight after 10 observations. 
# Epoch is a round of whole data flow through the network. Here we choose 100.
classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

# With the model fitted, we test the model on test data. Use a threshold of 0.5, to turn data into True(leaving) and False(stay) data.
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5)

# We use confusion_matrix to investigate the model performance on the test set.
cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix: ",cm)

# First, encode the variables. For instance, Geography France is encoded into (0,0) in the dummy variables, Gender male is 1. 
new_customer = np.array([[0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])
print("\nNew Customer: ",new_customer)

# Scale the data using the previously defined scaler for our training data
new_customer_scaled = sc.transform(new_customer)

# Request a prediction using the new data formatted as needed;
new_prediction = classifier.predict(new_customer_scaled)
new_prediction = (new_prediction > 0.5)
print("Customer exited: ",new_prediction)
