import cv2
import numpy as np
import matplotlib.pyplot as plt

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

yolo = cv2.dnn.readNet('/home/avik/SEM 6/VR/mini proj/yolov3.weights', '/home/avik/SEM 6/VR/mini proj/yolov3.cfg')
yolo.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
yolo.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
layer_names = yolo.getLayerNames()
output_layers = [layer_names[x[0] - 1] for x in yolo.getUnconnectedOutLayers()]

def detect_face(image):
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayscale_image, 1.2, 7)
    for (x,y,w,h) in faces:
        image = cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),5)
    cv2.imshow("detected_faces", image)
    cv2.waitKey(1)
    return faces

def my_yolo(image):
    yolo.setInput(cv2.dnn.blobFromImage(image, 0.00392, (416,416), (0,0,0), swapRB = True, crop = False))
    out = yolo.forward(output_layers)
    confidences = []
    boxes = []
    width = image.shape[1]
    height = image.shape[0]
    for curr_out in out:
        for detection in curr_out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.1 and class_id == 0:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = center_x - w / 2
                y = center_y - h / 2
                confidences.append(float(confidence))
                boxes.append([x, y, x + w, y + h])
    boxes = torch.tensor(boxes)
    confidences = torch.tensor(confidences)
    iou_threshold = 0.2
    indices = nms(boxes, confidences, iou_threshold)
    for i in indices:
        cv2.rectangle(image, (torch.round(boxes[i][0]), torch.round(boxes[i][1])), (torch.round(boxes[i][2]), torch.round(boxes[i][3])), (0, 0, 0), 2)
        cv2.putText(image, 'person', (torch.round(boxes[i][0]) - 10, torch.round(boxes[i][1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    boxes = [boxes[i] for i in indices]
    cv2.imshow("detected_humans", image)
    cv2.waitKey(1)
    return boxes, confidences

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 200, 3, 1, 0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(200, 100, 3, 1, 0)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(23 * 23 * 100, 50)
        self.fc2 = nn.Linear(50, 2)
        self.soft = nn.Softmax(dim=1)
        self.drop = nn.Dropout()

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(-1, 23 * 23 * 100)
        x = self.drop(x)
        x = F.relu(self.fc1(x))
        x = self.soft(self.fc2(x))
        return x

class ConvNet(nn.Module):

    def _init_(self):
        super(ConvNet, self)._init_()        
        self.ac = nn.ReLU()
        self.pool = nn.MaxPool2d(2)
        self.conv1 = nn.Conv2d(1,100,3)
        self.conv2 = nn.Conv2d(100,200,3)
        self.drop = nn.Dropout(0.2)
        self.fc1 = nn.Linear(200*23*23, 50)
        self.fc2 = nn.Linear(50, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.ac(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.ac(x)
        x = self.pool(x)
        x = x.view(-1,200*23*23)
        x = self.drop(x)
        x = self.fc1(x)
        x = self.fc2(x)
        return x

class data_convert(Dataset):
    def __init__(self, data):
        self.dataframe = data
    
    def __getitem__(self, ind):
        return self.dataframe[ind]
    
    def __len__(self):
        return len(self.dataframe)

PATH = '/home/avik/SEM 6/VR/mini proj/mask_classifier.pth'
net = Net()
net.load_state_dict(torch.load(PATH))
my_device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(my_device)

cap = cv2.VideoCapture('/home/avik/SEM 6/VR/mini proj/manan_with_mask.mp4')

while(True):
    ret, frame = cap.read()
    if ret == False:
        break
    boxes, confidences = my_yolo(frame)
    for i in range(len(boxes)):
        curr_img = frame[int(boxes[i][1]) : int(boxes[i][3]), int(boxes[i][0]) : int(boxes[i][2])]
        faces = detect_face(curr_img)
        valid = True 
        for j in range(len(faces)):
            curr_face = frame[int(faces[j][1]) : int(faces[j][3] + faces[j][1]), int(faces[j][0]) : int(faces[j][2] + faces[j][0])]
            curr_face_gray = cv2.cvtColor(curr_face, cv2.COLOR_BGR2GRAY)
            curr_face_gray_resize = cv2.resize(curr_face_gray, (100, 100))
            curr_input = []
            curr_input.append(curr_face_gray_resize)
            curr_data = []
            curr_data.append(curr_input)
            curr_data = torch.tensor(curr_data, dtype = torch.float32)
            curr_data = data_convert(curr_data)
            curr_data = torch.utils.data.DataLoader(curr_data, batch_size = 1, shuffle = True)
            for i, data in enumerate(curr_data):
                data = data.to(my_device)
                outputs = net(data)
            _, predicted = torch.max(outputs.data, 1)
            if predicted[0] == 0 and valid == True:
                valid = False
            if valid == False:
                break
        if valid == False:
            print("Don't open the door.")
        elif valid == True and len(faces) > 0:
            print("Open the door")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()