import torch.nn as nn
import numpy as np
import torch
import torch.optim as optim
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as Transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid

""""  Description of datasets
   -  MNIST  dataset is containing 10 digits numbers representing as numpy array (28,28)
   to corresponding label
   - and is already uploaded to torhvision to use 
"""
## Loading the dataset ##

train = datasets.MNIST(root="dataset", train=True,
                       transform=Transforms.ToTensor(), download=True)
test = datasets.MNIST(root="dataset", train=False,
                      transform=Transforms.ToTensor(), download=True)

train_set = DataLoader(dataset=train, batch_size=100, shuffle=True)

test_set = DataLoader(dataset=test, batch_size=100, shuffle=False)

# to keep in mind that daraloader is return a tuple (images , label )

image, label = next(iter(train_set))

print(image[0].shape, label.shape)


""""plotting some samples from dataset"""

image_ = make_grid(image[:12], nrow=12)
label_ = label[:12].numpy()
plt.xlabel(label_)
plt.imshow(np.transpose(image_.numpy(),(1,2,0)), cmap="gray")
plt.show()


""""intiallzie the model multi preceptonce"""


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


model = LinearNet()

for params in model.parameters():
    print(params.numel())


"""intialize instance of Loss funcrion and Optimizer"""

Optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()


""""
important step in pytorch framework is
you will need to creat your own Training Loop hand to fit Input dara 

"""


def Train(Epochs, train_data, test_data):
    train_loss = []
    test_corr = []
    train_corr = []
    test_loss = []
    for i in range(Epochs):
        train_corrected = 0
        test_corrected = 0
        for batch, (X_train, y_train) in enumerate(train_data):
            batch += 1
            X = X_train.reshape(X_train.shape[0], -1)
            y_Pred = model(X)
            loss = criterion(y_Pred, y_train)
            predicted_lable = torch.max(y_Pred.data, dim=1)[1]
            batch_corrected = (predicted_lable == y_train).sum()
            train_corrected += batch_corrected
            Optimizer.zero_grad()
            loss.backward()
            Optimizer.step()

            if batch % 200 == 0:
                acc = train_corrected.item()*100/100*batch
                print(f" epoch {i}, batch step {batch}, loss {loss},accuracy {acc}")

        train_loss.append(train_loss)
        train_corr.append(train_corrected)
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_data):

                X_val = X_test.reshape(X_test.shape[0], -1)
                y_val = model(X_val)
                y_lable = torch.max(y_val.data, dim=1)[1]
                test_corrected += (y_lable == y_test).sum()
        loss = criterion(y_val, y_test)
        test_loss.append(loss)
        test_corr.append(test_corrected)


if __name__ == '__main__':
    trainer = Train(10, train_set, test_set)
