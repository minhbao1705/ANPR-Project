import random

from Model import Lenet
from Transform import DataTransform
from Dataset import CharData
from torch.utils.data import DataLoader
import math
import torch
import argparse
import torch.nn as nn



device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

parser = argparse.ArgumentParser()
parser.add_argument("--path_train", type =str, default=None, help = "path train to data")
parser.add_argument("--path_val", type =str, default=None, help = "path val to data")
parser.add_argument("--batch_size",  type=int, default=16, help='total batch size for all GPUs')
parser.add_argument('--epochs', type=int, default=100)
opt = parser.parse_args()



dataset_train = CharData(opt.path_train, "train",transform= DataTransform(32, 45, 2))
dataset_val = CharData(opt.path_val, "val", transform= DataTransform(32,45,2))


dataloader_train = DataLoader(dataset= dataset_train,batch_size = opt.batch_size,  shuffle=True)
dataloader_val = DataLoader(dataset = dataset_val, batch_size= opt.batch_size, shuffle=False)



ALPHA_DICT = {0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'K', 9: 'L', 10: 'M', 11: 'N', 12: 'P',
              13: 'R', 14: 'S', 15: 'T', 16: 'U', 17: 'V', 18: 'X', 19: 'Y', 20: 'Z', 21: '0', 22: '1', 23: '2', 24: '3',
              25: '4', 26: '5', 27: '6', 28: '7', 29: '8', 30: '9', 31: "Background"}


model = Lenet().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters())


for e in range(opt.epochs):
    train_loss = 0.0
      # Optional when not using Model Specific layer
    for i, (images, labels) in enumerate(dataloader_train):

        images = images.to(device)
        labels = labels.to(device) #8

        targets = model(images).to(device)
         #8, 32
        loss = criterion(targets, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        if (i + 1) % 10 == 0:
            print(f'Epoch [{e + 1}/{opt.epochs}], Step [{i + 1}/{len(dataloader_train)}], Loss: {loss.item():.4f}')
    print(f"TOTAL TRAIN LOSS: {train_loss}")
    print("-"*30)

with torch.no_grad():
    n_correct = 0
    n_samples = 0
    n_class_correct = [0 for i in range(32)]
    n_class_samples = [0 for i in range(32)]
    for images, labels in dataloader_val:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        # max returns (value ,index)
        _, predicted = torch.max(outputs, 1)
        n_samples += labels.size(0)
        n_correct += (predicted == labels).sum().item()

        for i in range(len(labels)):
            label = labels[i]
            pred = predicted[i]
            if (label == pred):
                n_class_correct[label] += 1
            n_class_samples[label] += 1

    acc = 100.0 * n_correct / n_samples
    print(f'Accuracy of the network: {acc} %')

    for i in range(32):
        acc = 100.0 * n_class_correct[i] / (n_class_samples[i] + 0.001)
        print(f'Accuracy of {ALPHA_DICT[i]}: {acc} %')
print('Finished Training')
PATH = './lenet.pth'
torch.save(model.state_dict(), PATH)
