import os
import shutil
from typing import List
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .containts import *

def setup_train():
    NUM_CLASSES = 1
    vgg = VGG16(
        # weights=vgg16_weight_path,
        include_top=False,
        input_shape=IMG_SIZE + (3,)
    )

    vgg16 = Sequential()
    vgg16.add(vgg)
    vgg16.add(layers.Dropout(0.3))
    vgg16.add(layers.Flatten())
    vgg16.add(layers.Dropout(0.5))
    vgg16.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

    vgg16.layers[0].trainable = False

    vgg16.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(lr=1e-4),
        metrics=['accuracy']
    )
    vgg16.compile(loss='binary_crossentropy', optimizer=Adam(
        lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=["accuracy"])

    vgg16.summary()
    return vgg16
