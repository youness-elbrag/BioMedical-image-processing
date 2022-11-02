# BioMedical-image-processing
this repo about implement Deep Learning technic on Medical images using Pytorch Meta AI Framwork and tools analysis images 

* Chapter 1: fundational Numpy Operation 

    * Most used built-methods in Numpy are :
        - Reshape method

          ```python
              import numpy as np 
              np.random.seed(42) 
              matrix = np.random.rand(25)  ## keep in mind ndim must be equal  
              matrix.reshape(3,5).ndim ## output ==> 2D 
          ```
        - Slicing and indexing 

            ```python
              import numpy as np 
              np.random.seed(42) 
              matrix3d = np.random.rand(2,3,5)
              print(matrix3d)
              slice_3d = matrix3d[0,:2,1:3] 
              # slice in 2D is [start-row:end-row ,start-col:end-col]
              #slicing in 3D [start-axis:end-axis , start-row:end-row,start-col:end-col]
            ```