from utils import Image_Utils
import cv2 #type: ignore
import torch#type: ignore
from model import CNNModel
from dataset import CustomDataset
from torch.utils.data import DataLoader #type:ignore
import numpy as np#type:ignore
from torchvision import transforms#type:ignore
from tqdm import tqdm#type:ignore
from utils import Metrics_Utils as m
from sklearn.metrics import confusion_matrix #type:ignore
import random
import matplotlib.pyplot as plt #type:ignore

DEVICE = "mps"
BATCH_SIZE = 64
CH=4

transform = transforms.Compose([
            transforms.Resize((96, 96)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5]) 
        ])

dataset = CustomDataset(data_dir = "Dataset", ch=CH)
#test_dataset = CustomDataset(data_dir= "Dataset_Test", data_=[], transform=transform, ch=CH)

train_dataset, val_dataset, test_dataset, _ = dataset.split()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)



model = CNNModel(device_=DEVICE, ch=CH)
model.load_state_dict(torch.load('models/RGBT.pth'))

model.eval()
true_labels = []
predicted_labels = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

conf_matrix = confusion_matrix(true_labels, predicted_labels)
recall = m.calculate_recall(conf_matrix)
f1 = m.calculate_f1_score(conf_matrix)
precision = m.calculate_precision(conf_matrix)
accuracy = m.calculate_accuracy_(conf_matrix)


print(f"Confusion Matrix: {conf_matrix}" )
print(f"F1 score : {f1}, Recall : {recall}, Precision : {precision}, Acc : {accuracy}")


dataset = CustomDataset(data_dir = "Dataset", ch=3)
#test_dataset = CustomDataset(data_dir= "Dataset_Test", data_=[], transform=transform, ch=CH)

train_dataset, val_dataset, test_dataset, _ = dataset.split()

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)


model = CNNModel(device_=DEVICE, ch=3)
model.load_state_dict(torch.load('models/RGB.pth'))

model.eval()
true_labels = []
predicted_labels = []
with torch.no_grad():
    for inputs, labels in tqdm(test_loader):
        inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        true_labels.extend(labels.cpu().numpy())
        predicted_labels.extend(predicted.cpu().numpy())

conf_matrix = confusion_matrix(true_labels, predicted_labels)
recall = m.calculate_recall(conf_matrix)
f1 = m.calculate_f1_score(conf_matrix)
precision = m.calculate_precision(conf_matrix)
accuracy = m.calculate_accuracy_(conf_matrix)


print(f"Confusion Matrix: {conf_matrix}" )
print(f"F1 score : {f1}, Recall : {recall}, Precision : {precision}, Acc : {accuracy}")


