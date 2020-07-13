import cv2
import os
import numpy as np
from tqdm import tqdm
from typing import List
import imutils
import shutil
from keras.applications.vgg16 import VGG16, preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from .containts import *


def init_train_data():
    # create path for training data
    if not os.path.exists(MODEL_DATA_PATH):
        os.mkdir(MODEL_DATA_PATH)
    if not os.path.exists(DATA_PATH):
        os.mkdir(DATA_PATH)
    for d in [TEST_DATA_PATH, TRAIN_DATA_PATH, VAL_DATA_PATH]:
        if not os.path.exists(d):
            os.mkdir(d)
        for c in ["YES", "NO"]:
            if not os.path.exists(os.path.join(d, c)):
                os.mkdir(os.path.join(d, c))
    # create path for preprocessing data
    for d in ["TRAIN_CROP", "VAL_CROP", "TEST_CROP"]:
        crop_isolated_data_path = os.path.join(DATA_PATH, d)
        if not os.path.exists(crop_isolated_data_path):
            os.mkdir(crop_isolated_data_path)

        for c in ["NO", "YES"]:
            if not os.path.exists(os.path.join(crop_isolated_data_path, c)):
                os.mkdir(os.path.join(crop_isolated_data_path, c))

    for CLASS in filter(lambda x: not x.startswith('.'), os.listdir(BASE_PATH)):
        IMG_NUM = len(os.listdir(BASE_PATH + CLASS))
        for (n, FILE_NAME) in enumerate(os.listdir(BASE_PATH + CLASS)):
            img = BASE_PATH + CLASS + '/' + FILE_NAME
            if n < 5:
                shutil.copy(img, os.path.join(
                    TEST_DATA_PATH, CLASS.upper(), FILE_NAME))
            elif n < 0.8*IMG_NUM:
                shutil.copy(img, os.path.join(
                    TRAIN_DATA_PATH, CLASS.upper(), FILE_NAME))
            else:
                shutil.copy(img, os.path.join(
                    VAL_DATA_PATH, CLASS.upper(), FILE_NAME))


def train_data_generator() -> (List, List, List):

    train_datagen = ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=[0.5, 1.5],
        horizontal_flip=True,
        vertical_flip=True,
        preprocessing_function=preprocess_input
    )

    test_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input
    )

    train_generator = train_datagen.flow_from_directory(
        TRAIN_DIR,
        color_mode='rgb',
        target_size=IMG_SIZE,
        batch_size=32,
        class_mode='binary',
        seed=RANDOM_SEED
    )

    validation_generator = test_datagen.flow_from_directory(
        VAL_DIR,
        color_mode='rgb',
        target_size=IMG_SIZE,
        batch_size=16,
        class_mode='binary',
        seed=RANDOM_SEED
    )

    return train_generator, validation_generator, []


def load_data(dir_path, img_size=(100, 100)):
    """
    Load resized images as np.arrays to workspace
    """
    X = []
    y = []
    i = 0
    labels = dict()
    for path in tqdm(sorted(os.listdir(dir_path))):
        if not path.startswith('.'):
            isolated_data_path: str = os.path.join(dir_path, path)
            labels[i] = path
            for file in os.listdir(isolated_data_path):
                if not file.startswith('.'):
                    img = cv2.imread(isolated_data_path + '/' + file)
                    X.append(img)
                    y.append(i)
            i += 1
    X = np.array(X)
    y = np.array(y)
    print(f'{len(X)} images loaded from {dir_path} directory.')
    return X, y, labels


def crop_imgs(set_name, add_pixels_value=0):
    """
    Finds the extreme points on the image and crops the rectangular out of them
    """
    set_new = []
    for img in set_name:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        # threshold the image, then perform a series of erosions +
        # dilations to remove any small regions of noise
        thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.erode(thresh, None, iterations=2)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # find contours in thresholded image, then grab the largest one
        cnts = cv2.findContours(
            thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        c = max(cnts, key=cv2.contourArea)

        # find the extreme points
        extLeft = tuple(c[c[:, :, 0].argmin()][0])
        extRight = tuple(c[c[:, :, 0].argmax()][0])
        extTop = tuple(c[c[:, :, 1].argmin()][0])
        extBot = tuple(c[c[:, :, 1].argmax()][0])

        ADD_PIXELS = add_pixels_value
        new_img = img[extTop[1]-ADD_PIXELS:extBot[1]+ADD_PIXELS,
                      extLeft[0]-ADD_PIXELS:extRight[0]+ADD_PIXELS].copy()
        set_new.append(new_img)

    return np.array(set_new)


def save_new_images(x_set, y_set, folder_name):
    i = 0
    for (img, imclass) in zip(x_set, y_set):
        if imclass == 0:
            cv2.imwrite(folder_name+'NO/'+str(i)+'.jpg', img)
        else:
            cv2.imwrite(folder_name+'YES/'+str(i)+'.jpg', img)
        i += 1


def preprocess_imgs(set_name, img_size):
    """
    Resize and apply VGG-15 preprocessing
    """
    set_new = []
    for img in set_name:
        img = cv2.resize(
            img,
            dsize=img_size,
            interpolation=cv2.INTER_CUBIC
        )
        set_new.append(preprocess_input(img))
    return np.array(set_new)


def preprocess_data() -> (List, List, List, List):
    # use predefined function to load the image data into workspace
    X_train, y_train, labels = load_data(TRAIN_DATA_PATH, IMG_SIZE)
    X_test, y_test, _ = load_data(TEST_DATA_PATH, IMG_SIZE)
    X_val, y_val, _ = load_data(VAL_DATA_PATH, IMG_SIZE)

    # apply this for each set
    X_train_crop = crop_imgs(set_name=X_train)
    X_val_crop = crop_imgs(set_name=X_val)
    X_test_crop = crop_imgs(set_name=X_test)

    save_new_images(X_train_crop, y_train,
                    folder_name=f'{DATA_PATH}/TRAIN_CROP/')
    save_new_images(X_val_crop, y_val, folder_name=f'{DATA_PATH}/VAL_CROP/')
    save_new_images(X_test_crop, y_test, folder_name=f'{DATA_PATH}/TEST_CROP/')

    X_train_prep = preprocess_imgs(set_name=X_train_crop, img_size=IMG_SIZE)
    X_test_prep = preprocess_imgs(set_name=X_test_crop, img_size=IMG_SIZE)
    X_val_prep = preprocess_imgs(set_name=X_val_crop, img_size=IMG_SIZE)

    return X_val_prep, X_test_prep, y_val, y_test
