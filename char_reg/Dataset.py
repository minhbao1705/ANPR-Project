import math
import random
import Transform
import numpy as np
import torch.cuda
from torch.utils.data import Dataset
import os
import cv2
from random import sample

random.seed(10)

class CharData(Dataset):
    def __init__(self, path, phrase, transform):
        self.transform = transform
        self.phrase = phrase
        self.img_list, self.label_list = self.load_images_labels(path)
        self.n = len(self.img_list)

    def __getitem__(self, index):
        return self.img_list[index], self.label_list[index]

    def __len__(self):
        return self.n

    def load_images_labels(self, path):
        path = path.replace("\\","/")
        img_path = os.path.join(path, "images")
        lab_path = os.path.join(path, "labels")
        imgs = []
        label_list = []
        img_list = []
        with open(os.path.join(lab_path, "anno.txt"), "r") as f:
            f0 = f.readlines()
            for i in range(len(f0)):
                f0[i] = f0[i].replace("\n", "").replace("\t", "")
                label_list.append(f0[i][-1].upper())  # (lab name, lab)
                imgs.append(f0[i][:len(f0[i])-1])

        for i in imgs:

            abs_path_img = os.path.join(img_path, i).replace("\\","/")
            print(abs_path_img)
            img = self.transform(cv2.imread(abs_path_img),self.phrase)
            img_list.append(img)

        return img_list, normalize_label(label_list)
def normalize_label(labels):
    list_labels = []
    for i in labels:
        if i == "A":
            label = 0
        elif i == "B":
            label = 1
        elif i == "C":
            label = 2
        elif i == "D":
            label = 3
        elif i == "E":
            label = 4
        elif i == "F":
            label = 5
        elif i == "G":
            label = 6
        elif i == "H":
            label = 7
        elif i == "K":
            label = 8
        elif i == "L":
            label = 9
        elif i == "M":
            label = 10
        elif i == "N":
            label = 11
        elif i == "P":
            label = 12
        elif i == "R":
            label = 13
        elif i == "S":
            label = 14
        elif i == "T":
            label = 15
        elif i == "U":
            label = 16
        elif i == "V":
            label = 17
        elif i == "X":
            label = 18
        elif i == "Y":
            label = 19
        elif i == "Z":
            label = 20
        elif i == "0":
            label = 21
        elif i == "1":
            label = 22
        elif i == "2":
            label = 23
        elif i == "3":
            label = 24
        elif i == "4":
            label = 25
        elif i == "5":
            label = 26
        elif i == "6":
            label = 27
        elif i == "7":
            label = 28
        elif i == "8":
            label = 29
        elif i == "9":
            label = 30
        elif i == "@":
            label = 31
        else:
            label = -1
            ValueError("Don't match file")
        list_labels.append(label)
    return list_labels



