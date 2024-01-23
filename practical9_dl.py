# -*- coding: utf-8 -*-
""" Created on 26/02/2022 by Bhakti Chotalia """

from matplotlib import pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout

# load dataset
(trainX, trainY), (testX, testY) = mnist.load_data()

# reshape dataset to have a single channel
trainX = trainX.reshape((trainX.shape[0], 28, 28, 1))
testX = testX.reshape((testX.shape[0], 28, 28, 1))

# convert from integers to floats
trainX = trainX.astype('float32')
testX = testX.astype('float32')

# normalize to range 0-1
trainX /= 255.0
testX /= 255.0

# define cnn model
model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', kernel_initializer='he_uniform',
input_shape=(28, 28, 1)))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu', kernel_initializer='he_uniform'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='softmax'))

# compile and fit model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
metrics=['accuracy'])
model.fit(trainX, trainY, epochs=3, batch_size=32)

# evaluate model
model.evaluate(testX, testY)

# predict
image_index = int(input("Enter image index: "))
plt.imshow(testX[image_index].reshape(28,28),cmap='Greys')
predict = testX[image_index].reshape(28,28)
pred = model.predict(testX[image_index].reshape(1,28,28,1))

print("The prdeicted number is ",pred.argmax())
