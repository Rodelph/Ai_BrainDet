from tensorflow.keras.utils import normalize
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Activation, Dropout, Flatten, Dense
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
import numpy as np
import cv2 as cv
import os
from PIL import Image
from sklearn.model_selection import train_test_split

img_dir = 'static/Data_sets/'
yes_tumor_img = os.listdir(img_dir + 'yes/')
no_tumor_img = os.listdir(img_dir + 'no/')

dataset = []
label = []

INPUT_SIZE = 64

for i, image_name in enumerate(no_tumor_img):
    if image_name.split('.')[1] == 'jpg':
        img = cv.imread(img_dir + 'no/' + image_name)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(img))
        label.append(0)

for i, image_name in enumerate(yes_tumor_img):
    if image_name.split('.')[1] == 'jpg':
        img = cv.imread(img_dir + 'yes/' + image_name)
        img = Image.fromarray(img, 'RGB')
        img = img.resize((INPUT_SIZE, INPUT_SIZE))
        dataset.append(np.array(img))
        label.append(1)

dataset = np.array(dataset)
label = np.array(label)

x_train, x_test, y_train, y_test = train_test_split(dataset, label, test_size=0.2, random_state=0)

x_train = normalize(x_train, axis=1)
x_test = normalize(x_test, axis=1)

y_train = to_categorical(y_train, num_classes=2)
y_test = to_categorical(y_test, num_classes=2)

model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=(INPUT_SIZE, INPUT_SIZE, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), kernel_initializer='he_uniform'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(2))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train, y_train, batch_size=64, verbose=1, epochs=1000, validation_data=(x_test, y_test), shuffle=False)
model.save('TrainedFile/TrainFileTest.h5')
