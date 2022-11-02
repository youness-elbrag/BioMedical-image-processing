import numpy as np


seed = np.random.seed(24)

""""intialization matrix 2d and 3d with diffrent built-methods in Numpy"""
matrix = np.random.rand(5,5)
matrix_ = np.random.randn(5,4)

print(f"shape matrix 1 {matrix} other one {matrix_.shape}")

out = np.matmul(matrix,matrix_)
out_re = out.reshape(-1)
print(f"out of multiple is : {out.ndim} \n max value  out re  is : {out_re.max()}")


""""Slicing and indexing through 2d array and 3d """

matrix3d = np.random.rand(2,3,5)
print(matrix3d)

slice_3d = matrix3d[0,:2,1:3] ## here general slicing in 3D is  [start-axis:end-axis , start-row:end-row, start-col:end-col]
print(f"slicing inside 3d  {slice_3d} \n sum of across th row {slice_3d.sum(axis=0)} and sum across col is : {slice_3d.sum(axis=1)}")

exit()

slice_m = matrix[1:3,1:3] ### general sytnax of index and slice is [start-row:end-row ,start-col:end-col]
print(f"slice at row index to 1 and grab index col 2 : {slice_m}")

""""selection by condition in array of 2d """

con_if = matrix[matrix < 0.25]

print(f"condition return less than 0.25 : {con_if}")
