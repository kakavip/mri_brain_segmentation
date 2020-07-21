import os
import re
import time
import enum
import shutil
from typing import List
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.models import load_model, model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .containts import *

MODEL_FILE_FORMAT = "model_{}_at_{}.json"
WEIGHT_FILE_FORMAT = "weight_{}_at_{}.h5"
MODEL_FILE_FREFIX = "model_{}_at_"
WEIGHT_FILE_PREFIX = "weight_{}_at_"


class ModelTypes(enum.Enum):
    VGG: str = "VGG"
    RESNET_50: str = "RESNET_50"


def setup_train(_type: ModelTypes = ModelTypes.VGG, weight_path: str = None):
    NUM_CLASSES = 1

    base_model_switcher: dict = {
        ModelTypes.VGG: VGG16(
            weights=weight_path,
            include_top=False,
            input_shape=IMG_SIZE + (3,)
        ),
        ModelTypes.RESNET_50: ResNet50(
            weights=weight_path,
            include_top=False,
            input_shape=IMG_SIZE + (3,)
        )
    }

    base_model = base_model_switcher[_type]

    trainer = Sequential()
    trainer.add(base_model)
    trainer.add(layers.Dropout(0.3))
    trainer.add(layers.Flatten())
    trainer.add(layers.Dropout(0.5))
    trainer.add(layers.Dense(NUM_CLASSES, activation='sigmoid'))

    trainer.layers[0].trainable = False

    trainer.compile(
        loss='binary_crossentropy',
        optimizer=RMSprop(lr=1e-4),
        metrics=['accuracy']
    )
    trainer.compile(loss='binary_crossentropy', optimizer=Adam(
        lr=0.0003, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=["accuracy"])

    trainer.summary()
    return trainer


def save_model(trainer: Sequential, file_path: str, _type: ModelTypes = ModelTypes.VGG):

    model_json = trainer.to_json()
    time_stamp: int = int(time.time())

    model_file_path: str = f"{file_path}/{MODEL_FILE_FORMAT.format(_type.value.lower(), time_stamp)}"
    weight_file_path: str = f"{file_path}/{WEIGHT_FILE_FORMAT.format(_type.value.lower(), time_stamp)}"

    with open(model_file_path, "w") as json_file:
        json_file.write(model_json)

    trainer.save_weights(weight_file_path)


def load_latest_model(file_path: str, _type: ModelTypes = ModelTypes.VGG) -> Sequential:
    model_file_prefix: str = MODEL_FILE_FREFIX.format(_type.value.lower())
    weight_file_prefix: str = WEIGHT_FILE_PREFIX.format(_type.value.lower())

    file_path_list: List[str] = filter(lambda x: x.startswith(
        model_file_prefix) or x.startswith(weight_file_prefix), os.listdir(file_path))

    numbers: List[int] = map(lambda x: re.findall(
        "[0-9]+", x)[-1], file_path_list)

    max_number: int = max(numbers)

    model_file_path: str = f"{file_path}/{MODEL_FILE_FORMAT.format(_type.value.lower())}"
    weight_file_path: str = f"{file_path}/{WEIGHT_FILE_FORMAT.format(_type.value.lower())}"

    json_file = open(model_file_path, r)
    loaded_model_json = json_file.read()
    json_file.close()

    loaded_model: Sequential = model_from_json(loaded_model_json)
    loaded_model.load_weights(weight_file_path)

    return loaded_model
