import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import time
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
import os
import csv

from dataloader import *
from module import *

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# args
Train = True
Finetune = False
train_set = './data/train_val'
test_set = './data/test_val'
modelpath = './model'
pretrain_model = './model/best.pkl'
lr = 0.001
EPOCH = 1000
batch_size = 8  
use_mixup = False

if Train == True:
    train_data = Train_loader(train_set)
    train_loader = data.DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True)
    test_data = Train_loader(test_set)
    test_loader = data.DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False)

    if Finetune == True:
        model = torch.load(pretrain_model)
    else:
        model = Net1.to(device)
        
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(EPOCH):

        model.train()
        sum_loss = 0
        for images, labels in tqdm(train_loader):
            images = images.float().to(device)
            optimizer.zero_grad()  
            images = images.unsqueeze(1) # b*1*100*100*100
            output = model(images).squeeze()
            # output = torch.sigmoid(output)
            loss = criterion(output, labels.long())
            sum_loss += loss
            
            loss.backward()  
            optimizer.step()  

        print('epoch:{} train_loss:{}'.format(epoch,sum_loss/len(train_loader)))


        # test on test set
        if epoch%1 == 0:

            model.eval()
            sum_loss = 0
            for images, labels in test_loader:
                images = images.float().to(device)
                images = images.unsqueeze(1)
                output = model(images)
                # output = torch.sigmoid(output)
                sum_loss += criterion(output, labels.long())

            print('epoch:{} test_loss:{}'.format(epoch,sum_loss/len(test_loader)))


            torch.save(model, modelpath+'/latest.pkl')
            if epoch == 0:
                lowest_loss = sum_loss
            elif sum_loss < lowest_loss:
                lowest_loss = sum_loss
                torch.save(model, modelpath+'/best.pkl')




# def mixup_criterion(criterion, pred, y_a, y_b, lam):
#     return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)



# def mixup_data(x, y, alpha=0.5, use_cuda=False):
#     # 对数据的mixup 操作 x = lambda*x_i+(1-lamdda)*x_j
#     '''Returns mixed inputs, pairs of targets, and lambda'''
#     if alpha > 0:
#         lam = np.random.beta(alpha, alpha)
#     else:
#         lam = 1

#     batch_size = x.size()[0]
#     if use_cuda:
#         index = torch.randperm(batch_size).cuda()
#     else:
#         index = torch.randperm(batch_size)

#     mixed_x = lam * x + (1 - lam) * x[index, :]    # 此处是对数据x_i 进行操作
#     y_a, y_b = y, y[index]    # 记录下y_i 和y_j
#     return mixed_x, y_a, y_b, lam    # 返回y_i 和y_j 以及lambda

