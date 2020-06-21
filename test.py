import argparse
import json
import os
import numpy as np
import torch
import torchvision as tv
import torch.utils.data.dataloader as DataLoader

from model import dataloader
# from model.DnCNN import DnCNN
# from model import Resnet
from model import VoxNet
# from model import baseline
import pandas as pd
if __name__ == "__main__":
    config = json.load(open("config.json"))
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", default=config["GPU"], type=str, help="choose which DEVICE to use")
    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    test_set = dataloader.test_set()
    test_loader = DataLoader.DataLoader(test_set, batch_size=1, shuffle=False, num_workers=config["num_workers"]) 
    model = VoxNet.VoxNet(2).to(DEVICE)
    # Test the train_loader
    path = './model.pkl'
    model.load_state_dict(torch.load(path))
    model = model.eval()

    with torch.no_grad():
        # Test the test_loader
        test_loss = 0
        correct = 0
        idx = []
        Name = []
        Score = []
        
        for batch_idx, [data,name] in enumerate(test_loader):
            data= data.to(DEVICE)
            out = model(data)
            out = out.squeeze()
            Name.append(name[0])
            Score.append(out[1].item())

    test_dict = {'name':Name, 'predicted':Score}
    test_dict_df = pd.DataFrame(test_dict)
    print(test_dict_df)
    # path = './'
    # if not os.path.exists(path):
    #     os.makedirs(path)
    test_dict_df.to_csv('./submisson.csv', index=False)