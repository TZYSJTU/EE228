import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from torch.utils.data import Dataset

class VoxNet(torch.nn.Module):
    def __init__(self, num_classes, input_shape=(32, 32, 32)):
                 # weights_path=None,
                 # load_body_weights=True,
                 # load_head_weights=True):
        """
        VoxNet: A 3D Convolutional Neural Network for Real-Time Object Recognition.
        Modified in order to accept different input shapes.
        Parameters
        ----------
        num_classes: int, optional
            Default: 10
        input_shape: (x, y, z) tuple, optional
            Default: (32, 32, 32)
        weights_path: str or None, optional
            Default: None
        load_body_weights: bool, optional
            Default: True
        load_head_weights: bool, optional
            Default: True
        Notes
        -----
        Weights available at: url to be added
        If you want to finetune with custom classes, set load_head_weights to False.
        Default head weights are pretrained with ModelNet10.
        """
        super(VoxNet, self).__init__()
        self.preprocess = torch.nn.Sequential(
            # nn.Conv3d(2, 4, 3),
            nn.Conv3d(1, 8, 5, 2, padding=1),
            nn.LeakyReLU(),
            nn.BatchNorm3d(8),
            nn.Conv3d(8, 16, 9),
            nn.LeakyReLU(),
            nn.BatchNorm3d(16),
            nn.Conv3d(16, 16, 9),
            nn.LeakyReLU(),
            nn.BatchNorm3d(16),
        )
        self.body = torch.nn.Sequential(
            torch.nn.Conv3d(in_channels=16,
                            out_channels=32, kernel_size=5, stride=2),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.Dropout(p=0.2),
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(),
            nn.MaxPool3d(2),
            nn.Dropout(p=0.3)
        )

        # Trick to accept different input shapes
        x = self.body(torch.autograd.Variable(
            torch.rand((1, 16,33,33,33) )))
        first_fc_in_features = 1
        for n in x.size()[1:]:
            first_fc_in_features *= n

        self.head = torch.nn.Sequential(
            nn.Linear(first_fc_in_features, 64),
            nn.ReLU(),
            nn.Dropout(p=0.4),
            nn.Linear(64, num_classes),
            nn.Softmax()
        )

        # if weights_path is not None:
        #    weights = torch.load(weights_path)
        #    if load_body_weights:
        #        self.body.load_state_dict(weights["body"])
        #    elif load_head_weights:
        #        self.head.load_state_dict(weights["head"])

    def forward(self, x):
        x = self.preprocess(x)
        x = self.body(x)
        x = x.view(x.size(0), -1)
        x = self.head(x)
        return x

class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # 1*100*100*100
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),  # 8*32*32*32
            #nn.BatchNorm3d(num_features=8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 8*16*16*16
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=32, kernel_size=3,padding=1),  # 32*16*16*16
            #nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 32*8*8*8
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3,padding=1),  # 64*8*8*8
            #nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 64*4*4*4
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64*4*4*4, 64*4),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64*4, 16),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(16, 1),
            # nn.ReLU(),
        )        
        # dropout消去某些连接，比例为p
        self.dropout = nn.Dropout(p=0.5)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        # out_drop = self.fc1_drop(out)
 
        # out_drop = self.fc1_drop(out)

        # out_drop = self.fc1_drop(out)
        return out

class Net_1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # 1*100*100*100
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),  # 8*32*32*32
            #nn.BatchNorm3d(num_features=8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 8*16*16*16
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=32, kernel_size=3,padding=1),  # 32*16*16*16
            #nn.BatchNorm3d(num_features=32),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 32*8*8*8
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3,padding=1),  # 64*8*8*8
            #nn.BatchNorm3d(num_features=64),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 64*4*4*4
        )
        self.fc1 = nn.Sequential(
            nn.Linear(64*4*4*4, 64*4),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(64*4, 16),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(16, 2),
            # nn.ReLU(),
        )        
        # dropout消去某些连接，比例为p
        self.dropout = nn.Dropout(p=0.5)

        # self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)

        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        
        # out /= torch.sum(out, dim=1)
        # out[:,1] = out[:,1]/torch.sum(out, dim=1)
        # out = self.sigmoid(out)
        # out_drop = self.fc1_drop(out)
 
        # out_drop = self.fc1_drop(out)

        # out_drop = self.fc1_drop(out)
        return out

class Net11(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        # 1*100*100*100
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=8, kernel_size=3, stride=1, padding=1),  # 8*32*32*32
            #nn.BatchNorm3d(num_features=8),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 8*16*16*16
        )
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=8, out_channels=32, kernel_size=3,padding=1),  # 32*16*16*16
            #nn.BatchNorm3d(num_features=32),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 32*8*8*8
        )
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3,padding=1),  # 32*8*8*8
            #nn.BatchNorm3d(num_features=32),
            nn.Dropout(p=0.8),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),  # 32*4*4*4
        )
        # self.conv4 = nn.Sequential(
        #     nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3,padding=1),  # 32*8*8*8
        #     #nn.BatchNorm3d(num_features=32),
        #     nn.Dropout(p=0.8),
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=2, stride=2),  # 32*4*4*4
        # )
        # self.conv5 = nn.Sequential(
        #     nn.Conv3d(in_channels=32, out_channels=32, kernel_size=3,padding=1),  # 32*8*8*8
        #     #nn.BatchNorm3d(num_features=32),
        #     nn.Dropout(p=0.8),
        #     nn.ReLU(),
        #     nn.MaxPool3d(kernel_size=2, stride=2),  # 32*4*4*4
        # )
        self.fc1 = nn.Sequential(
            nn.Linear(32*4*4*4, 32*4),
            nn.Dropout(p=0.8),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(32*4, 16),
            nn.Dropout(p=0.8),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(16, 1),
            # nn.ReLU(),
        )        

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        # out = self.conv4(out)
        # out = self.conv5(out)
        
        out = out.view(out.size(0), -1)

        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        out = self.sigmoid(out)
        # out_drop = self.fc1_drop(out)
 
        # out_drop = self.fc1_drop(out)

        # out_drop = self.fc1_drop(out)
        return out

class Net2(nn.Module):
    def __init__(self):
        super(Net2, self).__init__()
        # 1*32*32*32
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        #32*16*16*16
        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        #128*8*8*8
        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=5, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )

        #256*2*2*2
        self.fc1 = nn.Linear(256*2*2*2, 256)
        # dropout消去某些连接，比例为p
        #self.fc1_drop = nn.Dropout(p=0)
        self.fc2 = nn.Linear(256, 64)
        self.fc3 = nn.Linear(64, 16)
        self.fc4 = nn.Linear(16, 2)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout3d(p=0.8)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = out.view(out.size(0), -1)
        out = self.relu(self.fc1(out))
        out = self.drop(out)
        #out = self.fc1_drop(out)
        out = self.relu(self.fc2(out))
        out = self.drop(out)
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        return out

class Net3(nn.Module):
    def __init__(self):
        super(Net3, self).__init__()

        self.fc1 = nn.Sequential(
            nn.Linear(32**3, 2048),
            nn.Dropout(p=0.8),
            nn.ReLU(),
        )
        self.fc2 = nn.Sequential(
            nn.Linear(2048, 128),
            nn.Dropout(p=0.8),
            nn.ReLU(),
        )
        self.fc3 = nn.Sequential(
            nn.Linear(128, 1),
            nn.Dropout(p=0.8),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = x.view(x.size(0), -1)
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)

        return out
