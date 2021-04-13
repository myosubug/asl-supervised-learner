# This Knn training is used to train ASL images classification
# CPSC 599.44 ML W2021
# Team Project

from PIL import Image
from numpy import asarray
from skimage.transform import resize
from skimage.feature import hog
from skimage.color import rgb2grey
from sklearn.model_selection import KFold, cross_val_score,train_test_split
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
import glob
import os
import numpy as np
import collections

x_test = []
y_test = []
x_train = []
y_train = []

'''
this part of the code is for downsampled data set with background
the data is in https://www.kaggle.com/grassknoted/asl-alphabet
and you will have to unpack the zip file from the download
and put it like below

"training"
   |
   |-- "data"
   |     |
   |   "asl_alphabet_train" <-- this is folder you have downloaded

also you will have to delete "del" and "space" folder from the dataset 
so that we onyl can have 27 classes     
'''

class_lookup = collections.defaultdict(int)
print("loading train data set")
# for importing training data
for img in glob.glob("dataset/asl_alphabet_train/**/*.jpg", recursive=True):
    img_name = os.path.basename(img)
    if class_lookup[img_name[0]] > 500:
        continue
    elif "nothing" in img_name and class_lookup["nothing"] > 500:
        continue
    else:
        opened = Image.open(img)
        into_array = asarray(opened)
        resized = resize(into_array, (64, 64, 3))
        x_train.append(resized)
        if "nothing" in img_name:
            y_train.append(0)
            class_lookup["nothing"] += 1
        else:
            y_train.append(ord(img_name[0]))
            class_lookup[img_name[0]] += 1
 
    
print("data loaded, checking loaded data status..")
#print(class_lookup)
print("spliting data")
np_data = np.array(x_train)
np_label = np.array(y_train)

m_samples = np_data.shape[0]
reshaped = np_data.reshape(m_samples, -1)

X_train, X_test, Y_train, Y_test = train_test_split(reshaped, np_label, stratify=np_label, test_size=0.25)
print("Start training..")

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, Y_train)
print("KNN test set accuracy: {:.3f}".format(knn.score(X_test, Y_test)))



'''
#this part of code is dataset with no background
#https://www.kaggle.com/ayuraj/asl-dataset trying dataset with no background

"training"
   |
   |-- "data"
   |     |
   |   "nobg" <-- this is folder you have downloaded and you have to rename it     

print("loading data set")
# put "data" folder in the same location as your knn.py or svm.py or cnn.py
# for importing testing data
for img in glob.glob("data/nobg/**/*.jpeg"):
    opened = Image.open(img)
    into_array = asarray(opened)
    resized = resize(into_array, (64, 64, 3))
    x_test.append(resized)
    img_name = os.path.basename(img).split("_")
    y_test.append(ord(img_name[1]))


print("data loaded")
print("spliting data")

np_data = np.array(x_test)
np_label = np.array(y_test)

m_samples = np_data.shape[0]
reshaped = np_data.reshape(m_samples, -1)

X_train, X_test, Y_train, Y_test = train_test_split(reshaped, np_label, stratify=np_label, test_size=0.25)

print("Start training..")

knn = KNeighborsClassifier(n_neighbors=15)
knn.fit(X_train, Y_train)
print("Knn test set accuracy: {:.3f}".format(knn.score(X_test, Y_test)))

'''