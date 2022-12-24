from keras import layers
from keras import optimizers
from keras import losses
from keras.models import Model
from keras.models import load_model
from keras.datasets import cifar10
from keras.utils import np_utils
from keras.utils import plot_model
import matplotlib.pyplot as plt
import numpy as np
import os

(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)
print(len(train_labels))
print(train_labels)
print(len(test_labels))
print(test_labels)
X_train = train_images.astype('float32') / 255
X_test = test_images.astype('float32') / 255
print(X_train.shape)
print(X_test.shape)
Y_train = np_utils.to_categorical(train_labels)
Y_test = np_utils.to_categorical(test_labels)

input_img = layers.Input(shape=(32, 32, 3))
vgg = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(input_img)
vgg = layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(vgg)
vgg = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='same')(vgg)
vgg = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(vgg)
vgg = layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(vgg)
vgg = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(vgg)
vgg = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(vgg)
vgg = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(vgg)
vgg = layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(vgg)
vgg = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same' )(vgg)
vgg = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(vgg)
vgg = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(vgg)
vgg = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(vgg)
vgg = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(vgg)
vgg = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(vgg)
vgg = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(vgg)
vgg = layers.Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding='same')(vgg)
vgg = layers.MaxPooling2D(pool_size=(2, 2), strides=(1, 1), padding='same')(vgg)
vgg = layers.Flatten()(vgg)
vgg = layers.Dense(4096, activation='relu')(vgg)
vgg = layers.Dropout(0.5)(vgg)
vgg = layers.Dense(4096, activation='relu')(vgg)
vgg = layers.Dropout(0.5)(vgg)
out_class = layers.Dense(10, activation='softmax')(vgg)
vgg_model = Model(input_img, out_class)

vgg_model.compile(optimizer=optimizers.Adam(), loss=losses.binary_crossentropy)
cm = vgg_model.fit(X_train, Y_train, epochs=1, batch_size=128, validation_data=(X_test, Y_test))
result = vgg_model.predict(X_test)
print(vgg_model.summary())
print(result.shape)
print(result[1,:])

vgg_model.save('e:/vgg_model.h5')
vgg_model.save_weights('e:/vgg_model_w.h5')
new_model = load_model('e:/vgg_model.h5')


