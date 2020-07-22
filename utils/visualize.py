from typing import List

import numpy as np
import matplotlib.pyplot as plt


def plot_samples(X, y, labels_dict, n=50):
    """
    Creates a gridplot for desired number of images (n) from the specified set
    """
    for index in range(len(labels_dict)):
        imgs = X[np.argwhere(y == index)][:n]
        j = 10
        i = int(n/j)

        plt.figure(figsize=(15, 6))
        c = 1
        for img in imgs:
            plt.subplot(i, j, c)
            plt.imshow(img[0])

            plt.xticks([])
            plt.yticks([])
            c += 1
        plt.suptitle('Tumor: {}'.format(labels_dict[index]))
        plt.show()


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=90)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    cm = np.round(cm, 2)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def show_image_process_duration(img, img_cnt, img_pnt, new_img):
    plt.figure(figsize=(15, 6))
    plt.subplot(141)
    plt.imshow(img)
    plt.xticks([])
    plt.yticks([])
    plt.title('Step 1. Get the original image')
    plt.subplot(142)
    plt.imshow(img_cnt)
    plt.xticks([])
    plt.yticks([])
    plt.title('Step 2. Find the biggest contour')
    plt.subplot(143)
    plt.imshow(img_pnt)
    plt.xticks([])
    plt.yticks([])
    plt.title('Step 3. Find the extreme points')
    plt.subplot(144)
    plt.imshow(new_img)
    plt.xticks([])
    plt.yticks([])
    plt.title('Step 4. Crop the image')
    plt.show()


if __name__ == "__main__":
    X_train: List = []
    y_train: List = []
    labels: List = []
    plot_samples(X_train, y_train, labels, 10)
