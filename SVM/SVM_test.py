# use support vector machine
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import accuracy_score
import torchvision.transforms as transforms
import os
import csv
import time
import pickle
import numpy as np
from tqdm import tqdm

time_start = time.time()
train_path = "data/train_val"
test_path = "data/test"

train_file = []
train_id = []

test_file = []
test_id = []

Labels = {}
# train_file 文件名+后缀
# train_id   文件名
for file_name in os.listdir(train_path):
    prefix = file_name.split('.')[0]
    train_file.append(file_name)
    train_id.append(prefix)

for file_name in os.listdir(test_path):
    prefix = file_name.split('.')[0]
    test_file.append(file_name)
    test_id.append(prefix)

with open("data/augmentation.csv", "r") as f:
    reader = csv.reader(f)
    for line in reader:
        if line[0]=='name':
            continue
        Labels[line[0]] = int(line[1])

###########################################
num = 0
label = []
for file_name in train_file:
    
    prefix = file_name.split('.')[0]
    label.append(Labels[prefix])
    tmp = np.load(os.path.join(train_path, file_name))
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


#############################
num = 0
label = []
for file_name in test_file:
    
    prefix = file_name.split('.')[0]
    #label.append(Labels[prefix])
    tmp = np.load(os.path.join(test_path, file_name))
    voxel = tmp['voxel']
    seg = tmp['seg']

    vocel_seg = voxel * seg
    vocel_seg = vocel_seg.flatten()
    vocel_seg = vocel_seg[np.newaxis,:]
    # central = img[34:66, 34:66, 34:66]
    # central = self.data_preproccess(central)
    if num == 0:
        test_X = vocel_seg
        num += 1
    else:
        num += 1
            
        test_X = np.concatenate((test_X,vocel_seg), axis=0)
# test_Y = np.array(label)

# ######################################
# with open('train.pkl', 'rb') as f:
#     train_X = pickle.load(f)
#     train_Y = pickle.load(f)
# with open('test.pkl', 'rb') as f:
#     test_X = pickle.load(f)
#     test_Y = pickle.load(f)

print(time.time()-time_start)
clf = svm.LinearSVC(penalty='l2', loss='squared_hinge', dual=False, \
    tol=0.2, C=2, multi_class='ovr',fit_intercept=True, intercept_scaling=1, \
    class_weight=None, verbose=1, random_state=None, max_iter=1000)
# LinearSVC(dual=False, tol= 0.2, C=5, max_iter=2000)
clf.fit(train_X, train_Y) # train

print(time.time()-time_start)

predictions = []
for i in range(train_X.shape[0]):
    # prediction
    pre = clf.predict([train_X[i]])
    predictions.append(pre)
train_accuracy=accuracy_score(train_Y,predictions)

# predictions = []
# for i in range(65):
#     # prediction
#     pre = clf.predict([test_X[i]])
#     predictions.append(pre)
# test_accuracy=accuracy_score(test_Y,predictions)

with open('./res.csv', "w", newline='') as f:
    writer = csv.writer(f)
    header = ['name', 'predicted']
    writer.writerows([header])
    i= 0
    for file_name in test_file:
        datahead = []
        prefix = file_name.split('.')[0]
        datahead.append(prefix)
        
        pre = clf.predict([test_X[i]])
        i+=1
        
        datahead.append(int(pre))
        writer.writerows([datahead])

print('done')


print('Training accuracy: %0.2f%%' % (train_accuracy*100))

print(time.time()-time_start)