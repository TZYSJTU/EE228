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


Files = {}

path = os.path.abspath('data/train_val')
with open("data/train_val.csv", "r") as f:
    reader = csv.reader(f)
    for instance in reader:
        if instance[0]=='name':
            continue
        file_name = instance[0]
        label = instance[1]
        Files[file_name] = label

with open("data/augmentation.csv", "w", newline='') as f:
    writer = csv.writer(f)
    header = ['name', 'label']
    writer.writerows([header])
    for file_name in Files:
        label = Files[file_name] 
        data = np.load(os.path.join(path, file_name+'.npz'))
        voxel = data['voxel']
        mask = data['seg']


        data = np.load(os.path.join(path, file_name+'.npz'))
        voxel = data['voxel']
        mask = data['seg']


        augmentation_index = 0
        header = [file_name, label]
        writer.writerows([header])
        ####### rotate ########
        augmentation_index += 1
        tmp_voxel = np.rot90(voxel, k=1, axes=(0, 1))
        tmp_mask = np.rot90(mask, k=1, axes=(0, 1))
        np.savez(os.path.join(path, file_name+'_'+str(augmentation_index)+'.npz'), voxel=tmp_voxel, seg=tmp_mask)
        header = [file_name+'_'+str(augmentation_index),label]
        writer.writerows([header])  
        ####### rotate ########
        augmentation_index += 1
        tmp_voxel = np.rot90(voxel, k=2, axes=(0, 1))
        tmp_mask = np.rot90(mask, k=2, axes=(0, 1))
        np.savez(os.path.join(path, file_name+'_'+str(augmentation_index)+'.npz'), voxel=tmp_voxel, seg=tmp_mask)
        header = [file_name+'_'+str(augmentation_index),label]
        writer.writerows([header]) 
        ####### rotate ########
        augmentation_index += 1
        tmp_voxel = np.rot90(voxel, k=3, axes=(0, 1))
        tmp_mask = np.rot90(mask, k=3, axes=(0, 1))
        np.savez(os.path.join(path, file_name+'_'+str(augmentation_index)+'.npz'), voxel=tmp_voxel, seg=tmp_mask)
        header = [file_name+'_'+str(augmentation_index),label]
        writer.writerows([header]) 
                ####### rotate ########
        augmentation_index += 1
        tmp_voxel = np.rot90(voxel, k=1, axes=(0, 2))
        tmp_mask = np.rot90(mask, k=1, axes=(0, 2))
        np.savez(os.path.join(path, file_name+'_'+str(augmentation_index)+'.npz'), voxel=tmp_voxel, seg=tmp_mask)
        header = [file_name+'_'+str(augmentation_index),label]
        writer.writerows([header]) 
                ####### rotate ########
        augmentation_index += 1
        tmp_voxel = np.rot90(voxel, k=2, axes=(0, 2))
        tmp_mask = np.rot90(mask, k=2, axes=(0, 2))
        np.savez(os.path.join(path, file_name+'_'+str(augmentation_index)+'.npz'), voxel=tmp_voxel, seg=tmp_mask)
        header = [file_name+'_'+str(augmentation_index),label]
        writer.writerows([header]) 
                ####### rotate ########
        augmentation_index += 1
        tmp_voxel = np.rot90(voxel, k=3, axes=(0, 2))
        tmp_mask = np.rot90(mask, k=3, axes=(0, 2))
        np.savez(os.path.join(path, file_name+'_'+str(augmentation_index)+'.npz'), voxel=tmp_voxel, seg=tmp_mask)
        header = [file_name+'_'+str(augmentation_index),label]
        writer.writerows([header]) 
                ####### rotate ########
        augmentation_index += 1
        tmp_voxel = np.rot90(voxel, k=1, axes=(1, 2))
        tmp_mask = np.rot90(mask, k=1, axes=(1, 2))
        np.savez(os.path.join(path, file_name+'_'+str(augmentation_index)+'.npz'), voxel=tmp_voxel, seg=tmp_mask)
        header = [file_name+'_'+str(augmentation_index),label]
        writer.writerows([header]) 
                ####### rotate ########
        augmentation_index += 1
        tmp_voxel = np.rot90(voxel, k=2, axes=(1, 2))
        tmp_mask = np.rot90(mask, k=2, axes=(1, 2))
        np.savez(os.path.join(path, file_name+'_'+str(augmentation_index)+'.npz'), voxel=tmp_voxel, seg=tmp_mask)
        header = [file_name+'_'+str(augmentation_index),label]
        writer.writerows([header]) 
                ####### rotate ########
        augmentation_index += 1
        tmp_voxel = np.rot90(voxel, k=3, axes=(1, 2))
        tmp_mask = np.rot90(mask, k=3, axes=(1, 2))
        np.savez(os.path.join(path, file_name+'_'+str(augmentation_index)+'.npz'), voxel=tmp_voxel, seg=tmp_mask)
        header = [file_name+'_'+str(augmentation_index),label]
        writer.writerows([header]) 
                
        




