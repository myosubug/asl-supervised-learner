# Jashan: gonna work on this

import glob
import os
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sn
import pandas as pd
from numpy import asarray
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import MaxPool2D
from keras.layers import BatchNormalization
from keras.layers import Flatten
from keras.layers import Conv2D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix


# Loads the full data set and returns X_train, X_test, y_train, y_test, y_test_noEncode
def loadData():
    data = []
    label =[]

    print("loading train data set")
    # for importing training data
    counter = 0
    for img in glob.glob("data/asl_alphabet_train/**/*.jpg", recursive=True):
        opened = Image.open(img)
        into_array = asarray(opened, dtype=np.float32)
        resized = resize(into_array, (64, 64, 3))
        data.append(resized)
        img_name = os.path.basename(img)
        if "nothing" in img_name:
            label.append(26)
        else:
            label.append(ord(img_name[0])-65)
        counter += 1

    print("Parsed images, splitting now ... ")
    np_data = np.array(data)
    np_label = np.array(label)

    X_train, X_test, y_train, y_test = train_test_split(np_data, np_label, stratify=np_label, test_size=0.2)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_test_noEncode = y_test

    X_train = X_train.reshape((X_train.shape[0], 64, 64, 3))
    X_test = X_test.reshape((X_test.shape[0], 64, 64, 3))

    # one-hot encode labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    return X_train, X_test, y_train, y_test, y_test_noEncode

# Loads the custom data set and returns X_train, X_test, y_train, y_test, y_test_noEncode
def loadCustomData():
    data = []
    label =[]

    print("loading custom data set")
    for img in glob.glob("custom_data/**/*.jpeg", recursive=True):
        opened = Image.open(img)
        into_array = asarray(opened, dtype=np.float32)
        resized = resize(into_array, (64, 64, 3))
        data.append(resized)
        img_name = os.path.basename(img)
        letter = img_name[0]
        if letter == 'A':
            label.append(0)
        elif letter == 'B':
            label.append(1)
        elif letter == 'C':
            label.append(2)
        elif letter == 'D':
            label.append(3)
        elif letter == 'E':
            label.append(4)
        elif letter == 'F':
            label.append(5)
        elif letter == 'G':
            label.append(6)
        elif letter == 'H':
            label.append(7)
        elif letter == 'I':
            label.append(8)
        else:
            print("Incorrect label")
        #label.append(ord(img_name[0])-65)

    print("Parsed images, splitting now ... ")
    np_data = np.array(data)
    np_label = np.array(label)

    X_train, X_test, y_train, y_test = train_test_split(np_data, np_label, stratify=np_label, shuffle=True, random_state=10, test_size=0.2)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_test_noEncode = y_test

    X_train = X_train.reshape((X_train.shape[0], 64, 64, 3))
    X_test = X_test.reshape((X_test.shape[0], 64, 64, 3))

    # one-hot encode labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)
    
    return X_train, X_test, y_train, y_test, y_test_noEncode
    


# Loads the no-background data set and returns X_train, X_test, y_train, y_test, y_test_noEncode
def loadNoBackground():
    print("Starting Now!")
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

    X_train, X_test, y_train, y_test = train_test_split(np_data, np_label, stratify=np_label, test_size=0.2)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    
    y_test = np.array(y_test)
    y_test_noEncode = y_test

    X_train = X_train.reshape((X_train.shape[0], 64, 64, 3))
    X_test = X_test.reshape((X_test.shape[0], 64, 64, 3))

    # one-hot encode labels
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    return X_train, X_test, y_train, y_test, y_test_noEncode

# Trains the model
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
    model.add(Dense(27, activation = 'softmax'))
    
    model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])

    results = model.fit(X_train, y_train, epochs=50, verbose=1)

    return model, results

def main():
    
    model = Sequential()

    # X_train, X_test, y_train, y_test, y_test_noEncode = loadCustomData() # set nodes in last layer to 9
    # X_train, X_test, y_train, y_test, y_test_noEncode = loadNoBackground() # set nodes in last layer to 26
    X_train, X_test, y_train, y_test, y_test_noEncode = loadData() # set nodes in last layer to 27

    model, results = trainModel(X_train, y_train)
    model.save("model.h5")
    
    # model = keras.models.load_model("model.h5")
        
    print(model.summary())

    # Evaluate on test set
    _, accuracy = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test)
    print("Accuracy on test set: {:.3f}".format(accuracy))

    # summarize history for accuracy
    plt.plot(results.history['accuracy'])
    plt.title('model accuracy: full set')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['training set'])
    plt.show()
    
    # confusion matrix
    matrix = confusion_matrix(y_test_noEncode, y_pred.argmax(axis=1))
    # Adjust index and columns to appropriate values if using CustomData or NoBackground
    df_cm = pd.DataFrame(matrix, index = [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ "], columns = [i for i in "ABCDEFGHIJKLMNOPQRSTUVWXYZ "])
    sn.set(font_scale=.5)
    sn.heatmap(df_cm, annot=True, cmap="Blues", annot_kws={"size": 8}) # font size

    plt.show()

main()