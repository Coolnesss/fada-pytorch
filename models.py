import torch
import torch.nn.functional as F
from torch import nn

'''
    Domain-Class Discriminator (see (3) in the paper)
    Takes in the concatenated latent representation of two samples from
    G1, G2, G3 or G4, and outputs a class label, one of [0, 1, 2, 3]
'''
class DCD(nn.Module):
    def __init__(self, H=64, D_in=784):
        super(DCD, self).__init__()
        self.fc1 = nn.Linear(D_in, H)
        self.fc2 = nn.Linear(H, H)
        self.out = nn.Linear(H, 4)

    def forward(self, x):
        out = F.relu(self.fc1(x))
        out = self.fc2(out)
        return F.softmax(self.out(out), dim=1)

''' Called h in the paper. Gives class predictions based on the latent representation '''
class Classifier(nn.Module):
    def __init__(self, D_in=64):
        super(Classifier, self).__init__()
        self.out = nn.Linear(D_in, 10)

    def forward(self, x):
        return F.softmax(self.out(x), dim=1)

''' 
    Creates latent representation based on data. Called g in the paper.
    Like in the paper, we use g_s = g_t = g, that is, we share weights between target
    and source representations.

    Model is as specified in section 4.1. See https://github.com/kuangliu/pytorch-cifar/blob/master/models/lenet.py
'''
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
    
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, 64)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)
        return out