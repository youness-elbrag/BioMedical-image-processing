import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import torchvision.transforms as Transforms
import torchvision.datasets as datasets
from torchvision.utils import make_grid
from helper import *
from neuralClasses import *
from multiprocessing import Process

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


""""
important step in pytorch framework is
you will need to creat your own Training Loop hand to fit Input dara

since that we try to exploer deffrent type of Neural netwokr on the same dataset and
we will use multiprocessing package to creat proccessing training loop

"""


def Train(Model,train_data, test_data, Epochs,reshape=False,Optimizer=False):
    train_loss = []
    test_corr = []
    train_corr = []
    test_loss = []
    for i in range(Epochs):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        train_corrected = 0
        test_corrected = 0
        for batch, (X_train, y_train) in enumerate(train_data):
            batch += 1
            if reshape == True:
                X = X_train
                y_Pred = Model(X)
            else:     
                X = X_train.reshape(X_train.shape[0], -1)
                y_Pred = Model(X)
            if Optimizer == True:    
                loss = Runer_Optimizer_.criterion(y_Pred, y_train)
                predicted_lable = torch.max(y_Pred.data, dim=1)[1]
                batch_corrected = (predicted_lable == y_train).sum()
                train_corrected += batch_corrected
                Runer_Optimizer_
            else:    
                loss = Runer_Optimizer.criterion(y_Pred, y_train)
                predicted_lable = torch.max(y_Pred.data, dim=1)[1]
                batch_corrected = (predicted_lable == y_train).sum()
                train_corrected += batch_corrected
                Runer_Optimizer

            if batch % 200 == 0:
                acc = train_corrected.item()*100/100*batch
                print(
                    f" epoch {i}, batch step {batch}, loss {loss},accuracy {acc}")

        train_loss.append(train_loss)
        train_corr.append(train_corrected)
        with torch.no_grad():
            for b, (X_test, y_test) in enumerate(test_data):
                if reshape == True:
                    X_val = X_test
                else: 
                    X_val = X_test.reshape(X_test.shape[0], -1)     
                y_val = Model(X_val)
                y_lable = torch.max(y_val.data, dim=1)[1]
                test_corrected += (y_lable == y_test).sum()
        loss = Runer_Optimizer.criterion(y_val, y_test)
        test_loss.append(loss)
        test_corr.append(test_corrected)
    return train_loss, train_corr, test_corr, test_loss


if __name__ == '__main__':

   """intialize model architecture based on NeuralClass.py"""
   model_1 = LinearNet()
   model_2 = Convolution(1,8,10)
   """intialize instance of Loss funcrion and Optimizer"""

   Runer_Optimizer = Optimizer(model_1 , lr=0.01)
   Runer_Optimizer_= Optimizer(model_2 , lr=0.01)
   """"plotting some samples from dataset and training"""

   Proceesing_Parallel_Training(Train(model_1, train_set, test_set,1),Train(model_2,train_set, test_set,1,reshape=True,Optimizer=True),Parallel_Training_GPU=True)
  
