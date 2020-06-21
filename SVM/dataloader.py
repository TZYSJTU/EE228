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
from torch.utils.data import Dataset
import os
import csv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# class Train_loader(Dataset):
#     def __init__(self, data_path):
#         self.path = data_path
#         self.train_file = []
#         self.train_id = []
        
#         # train_file 文件名及后缀
#         # train_id   文件名
#         for file in os.listdir(self.path):
#             prefix = file.split('.')[0]
#             self.train_file.append(file)
#             self.train_id.append(prefix)
            
#         self.train_label = {}
#         with open("data/train_val.csv", "r") as f:
#             reader = csv.reader(f)
#             for line in reader:
#                 if line[0] in self.train_id:
#                     self.train_label[line[0]] = int(line[1])

#         self.lenth = len(self.train_id)
#         self.toTensor = transforms.ToTensor()
#         self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

#     def __getitem__(self, index):
#         file_name = self.train_file[index]
#         file_id = self.train_id[index]

#         img = np.load(os.path.join(self.path, file_name))
#         img = img[0]*img
#         img = self.data_preproccess(img)

#         label = self.train_label[file_id]
#         # if label == 1:
#         #     label = np.array([1,0])
#         # else:
#         #     label = np.array([0,1])


#         return file_name, img, torch.tensor(label).float().to(device)

#     def __len__(self):
#         return self.lenth

#     def data_preproccess(self, data):
#         data = self.toTensor(data)
#         #data = self.normalize(data)
#         return data.to(device)


class Train_loader(Dataset):
    def __init__(self, data_path):
        self.path = data_path
        self.train_file = []
        self.train_id = []
        self.train_label = {}
        # train_file 文件名及后缀
        # train_id   文件名
        for file in os.listdir(self.path):
            prefix = file.split('.')[0]
            self.train_file.append(file)
            self.train_id.append(prefix)
      
        with open("data/train_val.csv", "r") as f:
            reader = csv.reader(f)
            for line in reader:
                if line[0]=='name':
                    continue
                self.train_label[line[0]] = int(line[1])

        self.lenth = len(self.train_id)
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __getitem__(self, index):
        file_name = self.train_file[index]
        file_id = self.train_id[index]

        tmp = np.load(os.path.join(self.path, file_name))
        img = tmp['voxel']
        mask = tmp['seg']

        img = img * mask
        # central = img[34:66, 34:66, 34:66]
        # central = self.data_preproccess(central)

        label = self.train_label[file_id]
        # if label == 1:
        #     label = np.array([1,0])
        # else:
        #     label = np.array([0,1])

        return img, torch.tensor(label).float().to(device)

    def __len__(self):
        return self.lenth

    def data_preproccess(self, data):
        data = self.toTensor(data)
        #data = self.normalize(data)
        return data.to(device)



class Test_loader(Dataset):
    def __init__(self, data_path):
        self.path = data_path
        self.train_file = []

        for file in os.listdir(self.path):
            self.train_file.append(file)

        self.lenth = len(self.train_file)
        self.toTensor = transforms.ToTensor()
        self.normalize = transforms.Normalize(mean=[0.5], std=[0.5])

    def __getitem__(self, index):
        file_name = self.train_file[index]

        tmp = np.load(os.path.join(self.path, file_name))
        img = tmp['voxel']
        mask = tmp['seg']
        img = img * mask
        return img, file_name

    def __len__(self):
        return self.lenth

    def data_preproccess(self, data):
        data = self.toTensor(data)
        return data.to(device)