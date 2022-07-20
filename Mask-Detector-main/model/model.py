import cv2
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from torchvision.ops import nms

class data_convert(Dataset):
    def __init__(self, data):
        self.dataframe = data
    
    def __getitem__(self, ind):
        return self.dataframe[ind]
    
    def __len__(self):
        return len(self.dataframe)

def detect_mask(curr_face, net, loaded_model):
    preprocess = transforms.Compose([transforms.ToPILImage(), transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor(), transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    arr = np.zeros((3, curr_face.shape[0], curr_face.shape[1]), dtype = int)
    for j in range(curr_face.shape[0]):
        for k in range(curr_face.shape[1]):
            for l in range(3):
                arr[l][j][k] = curr_face[j][k][l] 
    curr_face = arr
    curr_data = []
    curr_data.append(preprocess(np.uint8(curr_face)))
    curr_data = [t.numpy() for t in curr_data]
    curr_data = torch.tensor(curr_data, dtype = torch.float32)
    curr_data = data_convert(curr_data)
    curr_data = torch.utils.data.DataLoader(curr_data, batch_size = 1, shuffle = False)
    fin_data = []
    with torch.no_grad():
        for data in curr_data:
            data = data.to(my_device)
            outputs = net(data)
            outputs = outputs.to(my_device)
            fin_data.extend(outputs)
            fin_data = pd.DataFrame(fin_data)
            output = loaded_model.predict(fin_data)
            proba = loaded_model.predict_proba(fin_data)
            if proba[0][0] >= 0.2:
                return True
            return False
            # if output[0] == 0:
            #     return False
            # return True
            # if outputs[0][1] >= 0.5:
            #     return valid
            # else:
            #     return False
            # _, predicted = torch.max(outputs.data, 1)
            # if predicted[0] == 0 and valid == True:
            #     valid = False
            # return valid 
