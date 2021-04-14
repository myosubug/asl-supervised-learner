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

X_train = []
y_train = []

print("loading test data set")
# put "data" folder in the same location as your knn.py or svm.py or cnn.py
# for importing testing data
for img in glob.glob("data/custom_data/**/*.jpeg", recursive=True):
    opened = Image.open(img)
    into_array = asarray(opened, dtype=np.float32)
    resized = resize(into_array, (64, 64, 3))
    X_train.append(resized)
    img_name = os.path.basename(img)
    # can remove del
    y_train.append(ord(img_name[0]))
        
        
print("Parsed images, splitting now ... ")
np_data = np.array(X_train)
np_label = np.array(y_train)

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
model.add(Dense(9, activation = 'softmax'))

model.compile(optimizer = 'adam', loss = keras.losses.categorical_crossentropy, metrics = ["accuracy"])
model.fit(X_train, y_train, epochs=10, verbose=2)

print(model.summary())

# # Evaluate on test set
_, accuracy = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy on test set: {:.3f}".format(accuracy))