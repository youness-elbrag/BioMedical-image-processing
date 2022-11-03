import torch
import numpy as np

# Creat class object to store the output tensor operation 
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class EmptyOut:

    def __init__(self,dim ,outs):

        self.empty_1D = torch.empty(size=dim,out=outs)
          
        return print(f"final outpist of empyt list :{self.empty_1D}")
   

    def dim_n(self,dim):

        empty_2D=torch.empty(size=dim)

        return print(f"results of matrix size dim {empty_2D}")
         
""""here we will exploer some use cases of torch operation often used in DL """

# first let us start with convert between  numpy to tensor versa-vers

np_matrix = np.random.rand(2, 5)
to_tensor = torch.from_numpy(np_matrix)

print(f"numpy array {type(np_matrix)} \n tensor is : {type(to_tensor)} ")

""""convert data type of tensot """

to_tensor.dtype
### torch support defferent type of dara type to go for int [32,64]and float alos
dt_changed = to_tensor.type(torch.float64)
print(dt_changed)

"""intialize the random distrubtion """

normal_dis = torch.normal(mean=torch.arange(1.,2.),std=torch.arange(0,1))      
print(normal_dis.shape)

""""Matrix multiplication of tensors"""
# additional two 1D matrix
tensor_x = torch.rand(3,2)
outs_neural=tensor_x.unsqueeze(1)
print(tensor_x.shape)
print(outs_neural.shape)
tensor_y = torch.rand(2,2)
outs = torch.mm(tensor_x, tensor_y)
empty_tensor = EmptyOut(dim=(3,3),outs=outs)