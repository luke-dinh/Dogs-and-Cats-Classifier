from keras.datasets import cifar10
from keras.models import Sequential
from keras.layers import Dense, Activation, Convolution2D
from keras.layers import ZeroPadding2D, Dropout, MaxPooling2D, Flatten
from keras.callbacks import TensorBoard
import pickle

name = 'Cat_and_dog_new_model'

pickle_in = open("X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("y.pickle", "rb")
y = pickle.load(pickle_in)

model = Sequential()

model.add(Convolution2D(64, (3,3), strides = (1,1), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(128, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Convolution2D(256, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())

model.add(Dense(64))

model.add(Dense(1))
model.add(Activation('sigmoid'))

tensorboard = TensorBoard(log_dir="logs\{}".format(name))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

model.fit(X, y, batch_size = 32, epochs = 4, verbose = 1, validation_split = 0.2)







