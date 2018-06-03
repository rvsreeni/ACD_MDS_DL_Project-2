#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  2 18:35:04 2018

@author: macuser
"""

import numpy
from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.constraints import maxnorm
from keras.optimizers import SGD
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)

# load data
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# normalize inputs from 0-255 to 0.0-1.0
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train = X_train / 255.0
X_test = X_test / 255.0

# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]

# Create simple CNN model with two convolutional layers followed by max pooling and a 
# flattening out of the network to fully connected layers to make predictions.
# Convolutional input layer, 32 feature maps with a size of 3×3, a rectifier activation 
# function and a weight constraint of max norm set to 3.
# Dropout set to 20%.
# Convolutional layer, 32 feature maps with a size of 3×3, a rectifier activation function 
# and a weight constraint of max norm set to 3.
# Max Pool layer with size 2×2.
# Flatten layer.
# Fully connected layer with 512 units and a rectifier activation function.
# Dropout set to 50%.
# Fully connected output layer with 10 units and a softmax activation function.
# A logarithmic loss function is used with the stochastic gradient descent optimization 
# algorithm configured with a large momentum & weight decay start with a learning rate of 0.01.

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), padding='same', activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.2))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same', kernel_constraint=maxnorm(3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Create deep CNN model with below architecture
# Convolutional input layer, 32 feature maps with a size of 3×3 and a rectifier activation function.
# Dropout layer at 20%.
# Convolutional layer, 32 feature maps with a size of 3×3 and a rectifier activation function.
# Max Pool layer with size 2×2.
# Convolutional layer, 64 feature maps with a size of 3×3 and a rectifier activation function.
# Dropout layer at 20%.
# Convolutional layer, 64 feature maps with a size of 3×3 and a rectifier activation function.
# Max Pool layer with size 2×2.
# Convolutional layer, 128 feature maps with a size of 3×3 and a rectifier activation function.
# Dropout layer at 20%.
# Convolutional layer,128 feature maps with a size of 3×3 and a rectifier activation function.
# Max Pool layer with size 2×2.
# Flatten layer.
# Dropout layer at 20%.
# Fully connected layer with 1024 units and a rectifier activation function.
# Dropout layer at 20%.
# Fully connected layer with 512 units and a rectifier activation function.
# Dropout layer at 20%.
# Fully connected output layer with 10 units and a softmax activation function.

#model = Sequential()
#model.add(Conv2D(32, (3, 3), input_shape=(3, 32, 32), activation='relu', padding='same'))
#model.add(Dropout(0.2))
#model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#model.add(Dropout(0.2))
#model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#model.add(Dropout(0.2))
#model.add(Conv2D(128, (3, 3), activation='relu', padding='same'))
#model.add(MaxPooling2D(pool_size=(2, 2)))
#model.add(Flatten())
#model.add(Dropout(0.2))
#model.add(Dense(1024, activation='relu', kernel_constraint=maxnorm(3)))
#model.add(Dropout(0.2))
#model.add(Dense(512, activation='relu', kernel_constraint=maxnorm(3)))
#model.add(Dropout(0.2))
#model.add(Dense(num_classes, activation='softmax'))

# Compile model
epochs = 10
lrate = 0.01
decay = lrate/epochs
sgd = SGD(lr=lrate, momentum=0.9, decay=decay, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
print(model.summary())

# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, batch_size=32)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))