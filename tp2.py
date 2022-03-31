import numpy as np
from keras.datasets import mnist
from keras.utils import np_utils
from keras import backend as K
K.set_image_data_format('channels_first')
#fix random seed for reproducibility

seed = 7
np.random.seed(seed)

from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D


def get_data_mnist() :

    #load data
    (X_train, y_train),(X_test, y_test)= mnist.load_data()
    # reshape to be [samples][pixels][width][height]
    X_train = X_train.reshape(X_train.shape[0], 1, 28,28).astype('float32')
    X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
    #one hot encode outputs
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)
    num_classes = y_test.shape[1]
    return (X_train, y_train), (X_test, y_test), num_classes

def small_model():
    
    model = Sequential()
    model.add(Conv2D(64, (3, 3), input_shape=(1, 28, 28),
    activation='relu'))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["acc"])
    return model

(X_train, y_train), (X_test, y_test), num_classes=get_data_mnist()

model=small_model()
model.fit(X_train,y_train,validation_data=(X_test,y_test),epochs=10,batch_size=200)