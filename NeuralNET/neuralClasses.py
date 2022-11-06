import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F


class Convolution(nn.Module):
    def __init__(self,in_f , out_f , num_classes ):
        super(Convolution, self).__init__()
        self.fc_0 = nn.Conv2d(in_f, out_f, kernel_size=3,stride=1,padding=1)
        self.pool_0 = nn.MaxPool2d(kernel_size=(2,2),stride=1,padding=1)
        self.fc_1 = nn.Conv2d(out_f, 8, kernel_size=(2,2),stride=1,padding=1)
        self.pool_1 = nn.MaxPool2d(kernel_size=(2,2),stride=1,padding=1)
        self.fc_2 = nn.Linear(8*31*31, 84)
        self.out_fc = nn.Linear(84, num_classes)

    def forward(self,input_):
        input_ = F.relu(self.fc_0(input_))
        input_ = F.relu(self.pool_0(input_))
        input_ = F.relu(self.fc_1(input_))
        input_ = F.relu(self.pool_1(input_))
        input_ = input_.reshape(input_.shape[0],-1)
        input_ = F.relu(self.fc_2(input_))
        input_ = F.relu(self.out_fc(input_))
        return F.log_softmax(input_, dim=1) 

    @staticmethod
    def verify_shape(input_):
        return print(f"shape of pervouis layer {input_.shape}")    

    def count_parameters(self):
        for params in model.parameters():
            print(f"{params.numel()}")      


class LinearNet(nn.Module):
    def __init__(self, in_sz=784, out_in=10, layers=[120, 84]):
        super(LinearNet, self).__init__()
        self.fc_ = nn.Linear(in_sz, layers[0])
        self.fc_1 = nn.Linear(layers[0], layers[1])
        self.out = nn.Linear(layers[1], out_in)

    def forward(self, x):
        input_ = F.relu(self.fc_(x))
        input_ = F.relu(self.fc_1(input_))
        input_ = self.out(input_)
        
        return F.log_softmax(input_, dim=1)


class Optimizer:
    def __init__(self,model, lr ):
        self.optimizer = optim.Adam(model.parameters(), lr=lr)
        self.criterion = nn.CrossEntropyLoss()

    def __call__(self):
        Runer_Optimizer = Optimizer()
        Runer_Optimizer.zero_grad()
        self.criterion.backward()
        self.optimizer.step()

