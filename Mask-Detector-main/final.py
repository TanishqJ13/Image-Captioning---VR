import cv2
import numpy as np
import matplotlib.pyplot as plt
import time
import pandas as pd
import pickle

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

from yolo.person import *
from yolo import bbox
from yolo.util import *
from yolo.darknet import Darknet
from yolo.preprocess import prep_image, inp_to_image, letterbox_image
from detect_faces.face import *
from model.model import *

def arg_parse():
    parser = argparse.ArgumentParser(description = 'YOLO v3 Video Detection Module')
    parser.add_argument("--confidence", dest = "confidence", help = "Object Confidence to filter predictions", default = 0.5)
    parser.add_argument("--nms_thresh", dest = "nms_thresh", help = "NMS Threshhold", default = 0.4)
    parser.add_argument("--cfg", dest = 'cfgfile', help = "Config file", default = "yolo/cfg/yolov3.cfg", type = str)
    parser.add_argument("--weights", dest = 'weightsfile', help = "weightsfile", default = "yolo/yolov3.weights", type = str)
    parser.add_argument("--reso", dest = 'reso', help = "Input resolution of the network. Increase to increase accuracy. Decrease to increase speed", default = "416", type = str)
    return parser.parse_args()

cap = cv2.VideoCapture('/home/gaurav/Desktop/sem6/VR/before_midsem/mini_project/harsh_without_mask.mp4')
args = arg_parse()
confidence = float(args.confidence)
nms_thesh = float(args.nms_thresh)
num_classes = 80
bbox_attrs = 5 + num_classes
yolo = Darknet(args.cfgfile)
yolo.load_weights(args.weightsfile)
yolo.net_info["height"] = args.reso
inp_dim = int(yolo.net_info["height"])
assert inp_dim % 32 == 0 
assert inp_dim > 32
yolo.cuda()

PATH = '/home/gaurav/Desktop/sem6/VR/before_midsem/mini_project/model/classi4.pkl'
net = models.alexnet(pretrained = True)
my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(my_device)

loaded_model = pickle.load(open(PATH, 'rb'))
tot_cnt = 0
cnt = 0
out = cv2.VideoWriter('./output_harsh_without_mask.avi',cv2.VideoWriter_fourcc(*'MJPG'), 20, (352, 640))

while(True):
    ret, frame = cap.read()
    if ret == False:
        break
    curr_persons = detect_person(frame, yolo, inp_dim, confidence, num_classes, nms_thesh)
    img = frame
    for i in range(len(curr_persons)):
        if curr_persons[i][0][1] != -1:
            curr_person = frame[int(curr_persons[i][0][1]) : int(curr_persons[i][1][1]), int(curr_persons[i][0][0]) : int(curr_persons[i][1][0])]
            curr_faces = detect_face(curr_person)
            for j in range(len(curr_faces)):
                cnt += 1
                curr_face = curr_person[curr_faces[j][1] : curr_faces[j][1] + curr_faces[j][3], curr_faces[j][0] : curr_faces[j][0] + curr_faces[j][2]]
                curr_valid = detect_mask(curr_face, net, loaded_model)
                if curr_valid == True:
                    tot_cnt += 1
                else:
                    tot_cnt -= 1
                img = cv2.rectangle(img, (int(curr_persons[i][0][0]), int(curr_persons[i][0][1])), (int(curr_persons[i][1][0]), int(curr_persons[i][1][1])), (255, 0, 255), 2)
                img = cv2.rectangle(img, (int(curr_persons[i][0][0]) + curr_faces[j][0], int(curr_persons[i][0][1]) + curr_faces[j][1]), (int(curr_persons[i][0][0]) + curr_faces[j][2] + curr_faces[j][0], int(curr_persons[i][0][1]) + curr_faces[j][3] + curr_faces[j][1]), (0, 255, 255), 2)
        if curr_valid == True:
            img = cv2.putText(img, 'OPEN THE DOOR', (40, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
        else:
            img = cv2.putText(img, 'DON\'T OPEN THE DOOR', (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_4)
    out.write(img)
    # cv2.imshow("Detected Person and Detected Face", img)
    # cv2.waitKey(1)
if cnt == 0:
    print("Don't open the door because no one is outside the door.")
elif tot_cnt > 0:
    print("Open the door.")
else:
    print("Don't open the door because there is someone who has not wear the mask properly or maybe he/she has not wear the mask")
cap.release()
cv2.destroyAllWindows()
out.release()