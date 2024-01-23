# -*- coding: utf-8 -*-
""" Created on 29/01/2022 by Bhakti Chotalia """

#packages
from tflearn import DNN
from tflearn.layers.core import input_data, dropout, fully_connected 
from tflearn.layers.estimator import regression

#We define our input data X and expected results Y as a list of lists.
#Since neural networks in essence only deal with numerical values, we’ll transform our boolean expressions into numbers so that True=1 and 
#False=0
X = [[0,0], [0,1], [1,0], [1,1]]
Y = [[0], [1], [1], [0]]

#We define the input, hidden and the output layers.
input_layer = input_data(shape=[None, 2])
hidden_layer = fully_connected(input_layer , 2, activation='tanh') 
output_layer = fully_connected(hidden_layer, 1, activation='tanh') 

#We define the regressor that will perform backpropagation and train our network. 
#We’ll use Stochastic Gradient Descent as optimisation method and Binary Crossentropy as the loss function
regression = regression(output_layer , optimizer='sgd', loss='binary_crossentropy', learning_rate=5)
model = DNN(regression)

#We need to train the model. During this process, the regressor will try to optimise the loss function
model.fit(X, Y, n_epoch=5000, show_metric=True);

#predict all possible combinations and transform the outputs to booleans using simple list comprehension
expected = [i[0] > 0 for i in Y]
predicted = [i[0] > 0 for i in model.predict(X)]

print('\nExpected : ',expected)
print('\nPredicted : ',predicted)

#Weight Analysis
print('\nWeights in layer1: ', model.get_weights(hidden_layer.W), ', Biases in layer1: ', model.get_weights(hidden_layer.b))
print('\nWeights in layer2: ', model.get_weights(output_layer.W), ', Biases in layer2: ', model.get_weights(output_layer.b))
