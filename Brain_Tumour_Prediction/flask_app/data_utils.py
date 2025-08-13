import os
import cv2
import imutils
import numpy as np
import matplotlib.pyplot as plt
from os import listdir
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

def crop_brain_contour(image, plot=False):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)

    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        return image

    c = max(cnts, key=cv2.contourArea)
    extLeft = tuple(c[c[:, :, 0].argmin()][0])
    extRight = tuple(c[c[:, :, 0].argmax()][0])
    extTop = tuple(c[c[:, :, 1].argmin()][0])
    extBot = tuple(c[c[:, :, 1].argmax()][0])

    new_image = image[extTop[1]:extBot[1], extLeft[0]:extRight[0]]
    
    if plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(image)
        plt.axis('off')
        plt.title('Original Image')
        plt.subplot(1, 2, 2)
        plt.imshow(new_image)
        plt.axis('off')
        plt.title('Cropped Image')
        plt.show()

    return new_image

def load_data(dir_list, image_size):
    X, y = [], []
    image_width, image_height = image_size
    for directory in dir_list:
        label = 1 if directory.endswith('yes') else 0
        for filename in listdir(directory):
            image_path = os.path.join(directory, filename)
            image = cv2.imread(image_path)
            if image is None:
                continue
            image = crop_brain_contour(image)
            image = cv2.resize(image, (image_width, image_height))
            image = image / 255.0
            X.append(image)
            y.append([label])
    X, y = np.array(X), np.array(y)
    X, y = shuffle(X, y)
    return X, y

def plot_sample_images(X, y, n=50):
    for label in [0, 1]:
        images = X[np.argwhere(y == label)][:n]
        cols, rows = 10, n // 10
        plt.figure(figsize=(20, 10))
        for i, image in enumerate(images):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(image[0])
            plt.axis('off')
        label_text = "Yes" if label == 1 else "No"
        plt.suptitle(f"Brain Tumor: {label_text}")
        plt.show()

def split_data(X, y, test_size=0.3):
    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=test_size)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5)
    return X_train, y_train, X_val, y_val, X_test, y_test
