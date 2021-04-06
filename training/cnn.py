# Jashan: gonna work on this

import re
import sys
import glob
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from numpy import asarray
from skimage.transform import resize
from skimage.feature import hog
from skimage.color import rgb2grey
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from tensorflow import keras
from keras.layers import Input
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPooling2D
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split


#This is for the larger data set
def loadData():
    X_test = []
    y_test = []
    X_train = []
    y_train = []

    print("loading test data set")
    # put "data" folder in the same location as your knn.py or svm.py or cnn.py
    # for importing testing data
    for img in glob.glob("data/asl_alphabet_test/*.jpg"):
        opened = Image.open(img)
        into_array = asarray(opened, dtype=np.float32)
        resized = resize(into_array, (64, 64, 3))
        X_test.append(resized)
        img_name = os.path.basename(img)
        if "del" in img_name:
            y_test.append(26)
        elif "nothing" in img_name:
            y_test.append(27)
        elif "space" in img_name:
            y_test.append(28)
        else:
            y_test.append(ord(img_name[0])-65)

    print("loading train data set")
    # for importing training data
    counter = 0
    for img in glob.glob("data/asl_alphabet_train/**/*.jpg", recursive=True):
        opened = Image.open(img)
        into_array = asarray(opened, dtype=np.float32)
        resized = resize(into_array, (64, 64, 3))
        X_train.append(resized)
        img_name = os.path.basename(img)
        # can remove del
        if "del" in img_name:
            y_train.append(26)
        elif "nothing" in img_name:
            y_train.append(27)
        # can remove space
        elif "space" in img_name:
            y_train.append(2)
        else:
            y_train.append(ord(img_name[0])-65)
        counter += 1

    return X_train, X_test, y_train, y_test


# Loads the no-background data set and returns X_train, X_test, y_train, y_test
def loadNoBackground():
    print("Starting Now!")
    # This is for the smaller data set
    data = []
    label = []

    print("Getting images ...")
    for img in glob.glob("data-nobackground/**/*.jpeg"):
        opened = Image.open(img)
        into_array = resize(asarray(opened), (64, 64, 3))
        data.append(into_array)
        # Bug: Putting in 123 labels instead of 36
        img_name = os.path.basename(img).split("_")
        label.append(ord(img_name[1])-97)

    print("Parsed images, splitting now ... ")
    np_data = np.array(data)
    np_label = np.array(label)

    X_train, X_test, y_train, y_test = train_test_split(np_data, np_label, stratify=np_label, test_size=0.25)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    X_train = X_train.reshape((X_train.shape[0], 64, 64, 3))
    X_test = X_test.reshape((X_test.shape[0], 64, 64, 3))

    # one-hot encode labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test

# Trains the model with the no background data set
def trainModel(X_train, y_train):
    print("Training now ...")

    model = Sequential()
    
    model.add(Conv2D(64, kernel_size = 3, padding = 'same', activation = 'relu', input_shape = (64,64,3)))
    model.add(Conv2D(32, kernel_size = 3, padding = 'same', strides = 2, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(64, kernel_size = 3, padding = 'same', strides = 2, activation = 'relu'))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(Conv2D(256, kernel_size = 3, padding = 'same', strides = 2 , activation = 'relu'))
    model.add(MaxPool2D(3))
    model.add(BatchNormalization())
    model.add(Flatten())
    model.add(Dropout(0.25))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dense(26, activation = 'softmax'))
    
    model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])

    results = model.fit(X_train, y_train, epochs=10, verbose=2)

    return model

# Usage:
#   python cnn.py mode pathname
#   where mode = "save" or "load" and pathname is the pathname to the model to save to / load from
def main():
    
    pathname = ""
    train = False

    if len(sys.argv) == 1:
        pathname = "model.h5"
        train = True
    
    if len(sys.argv) == 2:
        pathname = "model"
        if(sys.argv[1]) == "save":
            train = True
        elif(sys.argv[1]) == "load":
            train = False
        else:
            print("Invalid mode, default to save")
    
    if len(sys.argv) == 3:
        pathname = sys.argv[2]
        if(sys.argv[1]) == "save":
            train = True
        elif(sys.argv[1]) == "load":
            train = False
        else:
            print("Invalid mode, default to save")
            train = True

    model = Sequential()

    X_train, X_test, y_train, y_test = loadData()

    if train:
        model = trainModel(X_train, y_train)
        model.save("model.h5")
    else:
        model = keras.models.load_model("model.h5")

    print(model.summary())

    # # Evaluate on test set
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print("Accuracy on test set: {:.3f}".format(accuracy))

    # # # summarize history for accuracy
    # plt.plot(results.history['accuracy'])
    # plt.title('model accuracy')
    # plt.ylabel('accuracy')
    # plt.xlabel('epoch')
    # plt.legend(['training set'])
    # plt.show()

main()