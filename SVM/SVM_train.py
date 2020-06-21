from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
import os
import csv
import time
import pickle
import random
import numpy as np
from tqdm import tqdm
time_start = time.time()


path = "data/train_val"

Files = []
Labels = {}
with open("data/augmentation.csv", "r") as f:
    reader = csv.reader(f)
    for line in reader:
        if line[0]=='name':
            continue
        Labels[line[0]] = int(line[1])
        Files.append(line[0])

random.shuffle(Files)
length = len(Labels)
train_num = int(length*0.8)
test_num = int(length*0.2)
train_file = Files[:train_num]
test_file = Files[-test_num:]
print('train_data:',train_num)
print('test_data:',test_num)
###########################################
num = 0
label = []
for file_name in train_file:
    label.append(Labels[file_name])
    tmp = np.load(os.path.join(path, file_name+'.npz'))
    voxel = tmp['voxel']
    seg = tmp['seg']

    vocel_seg = voxel * seg
    vocel_seg = vocel_seg.flatten()
    vocel_seg = vocel_seg[np.newaxis,:]
    # central = img[34:66, 34:66, 34:66]
    # central = self.data_preproccess(central)
    if num == 0:
        train_X = vocel_seg
        num += 1
    else:
        num += 1
            
        train_X = np.concatenate((train_X,vocel_seg), axis=0)
train_Y = np.array(label)
print(time.time()-time_start)

################################
#############  SVM  ############
clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=False, \
    tol=0.1, C=1, multi_class='ovr',fit_intercept=True, intercept_scaling=2, \
    class_weight=None, verbose=1, random_state=None, max_iter=1000)
# LinearSVC(dual=False, tol= 0.2, C=5, max_iter=2000)
clf.fit(train_X, train_Y) # train
print(time.time()-time_start)

###########################
########### test ##########
test_Y = []
predictions=[]
for file_name in test_file:
    label = Labels[file_name]
    test_Y.append(label)
    tmp = np.load(os.path.join(path, file_name+'.npz'))
    voxel = tmp['voxel']
    seg = tmp['seg']

    vocel_seg = voxel * seg
    vocel_seg = vocel_seg.flatten()
    vocel_seg = vocel_seg[np.newaxis,:]
    # central = img[34:66, 34:66, 34:66]
    # central = self.data_preproccess(central)

    # prediction
    pre = clf.predict(vocel_seg)
    predictions.append(int(pre))
    
# train_accuracy=accuracy_score(train_Y,predictions)

# predictions = []
# for i in range(65):
#     # prediction
#     pre = clf.predict([test_X[i]])
#     predictions.append(pre)
test_accuracy=accuracy_score(test_Y,predictions)

# print('Training accuracy: %0.2f%%' % (train_accuracy*100))
print('Testing accuracy: %0.2f%%' % (test_accuracy*100))
print(time.time()-time_start)