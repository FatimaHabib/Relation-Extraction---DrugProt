"""Define the neural network, loss function"""
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset

from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence,pad_packed_sequence
import torch.nn as nn
import torch
################
class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels = 1,out_channels= 40,kernel_size = 2,stride = 1),##stride= 1,kernel_size = 5),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(40, 60, kernel_size=2, stride=1, padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2))
        self.drop_out = nn.Dropout()
        self.fc1 = nn.Linear( 60*76*51,1000)##n-channels * size of the maxpool layer
        self.fc2 = nn.Linear(1000, 14)
        
#############################################"
    def forward(self, x):
        #print("Input shape before squeeze:",x.shape)
        x= x.unsqueeze(1)
        #print("Input shape",x.shape)
        out = self.layer1(x)
        #print("output shape after first layer", out.shape)
        out = self.layer2(out)
        #print("output shape after second layer",out.shape)
        out = out.reshape(out.shape[0], -1)
        #out= out.view(batch_size,-1)
        #print("Output shape after reshaping",out.shape)
        out = self.drop_out(out)
        #print("The output shape after dropout",out.shape)
        out = self.fc1(out)
        out = self.fc2(out)
        ##print(out.shape)
        return out
