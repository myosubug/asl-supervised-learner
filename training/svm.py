# Sean: gonna work on thisfrom PIL import Imagefrom numpy import asarrayfrom sklearn.model_selection import train_test_splitimport globimport osimport numpy as npx_test = []y_test = []x_train = []y_train = []# put "data" folder in the same location as your knn.py or svm.py or cnn.py# for importing testing datafor img in glob.glob("data/asl_alphabet_test/*.jpg"):    opened = Image.open(img)    into_array = asarray(opened)    x_test.append(into_array)    img_name = os.path.basename(img)    if "del" in img_name:        y_test.append(0)    elif "nothing" in img_name:        y_test.append(1)    elif "space" in img_name:        y_test.append(2)    else:        y_test.append(ord(img_name[0]))# for importing training datafor img in glob.glob("data/asl_alphabet_train/**/*.jpg", recursive=True):    opened = Image.open(img)    into_array = asarray(opened)    x_train.append(into_array)    img_name = os.path.basename(img)    if "del" in img_name:        y_train.append(0)    elif "nothing" in img_name:        y_train.append(1)    elif "space" in img_name:        y_train.append(2)    else:        y_test.append(ord(img_name[0]))np_data = np.array(x_test + x_train)np_label = np.array(y_test + y_train)X_train, X_test, Y_train, Y_test = train_test_split(np_data, np_label, stratify=np_label, test_size=0.2)