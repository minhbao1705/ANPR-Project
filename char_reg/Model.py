import torch.nn as nn
import torch.nn.functional as F



class Lenet(nn.Module):
    def __init__(self):
        super(Lenet, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size= 5, stride = 1, padding = 0)
        self.batchnorm1 = nn.BatchNorm2d(6)
        self.mpool1 = nn.MaxPool2d(kernel_size= 2, stride= 2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0)
        self.batchnorm2 = nn.BatchNorm2d(16)
        self.mpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.fc = nn.Linear(400, 120)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 32)

    def forward(self, x):
        #layer1
        out = self.conv1(x)
        out = F.relu(self.batchnorm1(out))
        out = self.mpool1(out)
        #layer2
        out = self.conv2(out)
        out = F.relu(self.batchnorm2(out))
        out = self.mpool2(out)
        #FCN
        out = out.reshape(out.size(0), -1)
        out = F.relu(self.fc(out))
        out = F.relu(self.fc1(out))
        out = self.fc2(out)
        return out


