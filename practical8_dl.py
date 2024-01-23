# -*- coding: utf-8 -*-
""" Created on 04/03/2022 by Bhakti Chotalia """

import matplotlib.pyplot as plt
import numpy as np
from keras.models import Model
from keras.layers import Dense, Input
from keras.datasets import mnist

def plot_autoencoder_outputs(autoencoder, n, dims):
  decoded_imgs = autoencoder.predict(x_test)
  # number of example digits to show
  n = 5
  plt.figure(figsize=(10, 4.5))
  for i in range(n):
      # plot original image
      ax = plt.subplot(2, n, i + 1)
      plt.imshow(x_test[i].reshape(*dims))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i==2:
        ax.set_title('Original Images')
      # plot reconstruction
      ax = plt.subplot(2, n, i + 1 + n)
      plt.imshow(decoded_imgs[i].reshape(*dims))
      plt.gray()
      ax.get_xaxis().set_visible(False)
      ax.get_yaxis().set_visible(False)
      if i==2:
        ax.set_title('Reconstructed Images')
  plt.show()
  
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0
x_train = x_train.reshape((len(x_train), np.prod(x_train.shape[1:])))
x_test = x_test.reshape((len(x_test), np.prod(x_test.shape[1:])))
print(x_train.shape)
print(x_test.shape)

input_size = 784 #784 numbers between [0, 1] (28x28 pixel)
hidden_size = 128 #128 nodes in the hidden layer - stacked autoencoder
code_size = 32 #code size is 32 (number of nodes in the middle layer. Smaller size results in more compression.)
input_img = Input(shape=(input_size,))

#encoding
hidden_1 = Dense(hidden_size, activation='relu')(input_img)
code = Dense(code_size, activation='relu')(hidden_1)

#decoding
hidden_2 = Dense(hidden_size, activation='relu')(code)
output_img = Dense(input_size, activation='sigmoid')(hidden_2)

autoencoder = Model(input_img, output_img)
print(autoencoder.summary())
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
autoencoder.fit(x_train, x_train, epochs=5)
plot_autoencoder_outputs(autoencoder, 5, (28, 28))
