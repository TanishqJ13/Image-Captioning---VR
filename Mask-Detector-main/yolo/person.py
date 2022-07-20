from __future__ import division
import time
import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
from yolo.util import *
from yolo.darknet import Darknet
from yolo.preprocess import prep_image, inp_to_image, letterbox_image
import pandas as pd
import random 
import pickle as pkl
import argparse

def prep_image(img, inp_dim):
    orig_im = img
    dim = orig_im.shape[1], orig_im.shape[0]
    img = (letterbox_image(orig_im, (inp_dim, inp_dim)))
    img_ = img[:,:,::-1].transpose((2,0,1)).copy()
    img_ = torch.from_numpy(img_).float().div(255.0).unsqueeze(0)
    return img_, orig_im, dim

def write(x):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    if int(x[-1]) != 0 or (c1[0] - c2[0]) * (c1[1] - c2[1]) == 0:
        return (-1, -1), (-1, -1)
    return c1, c2

def detect_person(frame, model, inp_dim, confidence, num_classes, nms_thesh):
    img, orig_im, dim = prep_image(frame, inp_dim)
    im_dim = torch.FloatTensor(dim).repeat(1,2)                        
    im_dim = im_dim.cuda()
    img = img.cuda()
    with torch.no_grad():   
        output = model(Variable(img), True)
    output = write_results(output, confidence, num_classes, nms = True, nms_conf = nms_thesh)
    if type(output) == int:
        curr = []
        return curr
    im_dim = im_dim.repeat(output.size(0), 1)
    scaling_factor = torch.min(inp_dim/im_dim,1)[0].view(-1,1)
    output[:,[1,3]] -= (inp_dim - scaling_factor*im_dim[:,0].view(-1,1))/2
    output[:,[2,4]] -= (inp_dim - scaling_factor*im_dim[:,1].view(-1,1))/2
    output[:,1:5] /= scaling_factor
    for i in range(output.shape[0]):
        output[i, [1,3]] = torch.clamp(output[i, [1,3]], 0.0, im_dim[i,0])
        output[i, [2,4]] = torch.clamp(output[i, [2,4]], 0.0, im_dim[i,1])
    curr = list(map(lambda x: write(x), output))
    return curr