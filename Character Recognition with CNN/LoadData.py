import numpy as np
import matplotlib.pyplot as plt
from keras.utils import HDF5Matrix

def Generator(hdf5_file,batch_size):
    X = HDF5Matrix(hdf5_file,"/Data/X")
    Y = HDF5Matrix(hdf5_file,"/Data/Y")
    
    print("HDFMatrix loaded")

    #return X,Y
    size = X.end
    idx = 0

    while True:
        last_batch = idx+batch_size >size
        end = idx + batch_size if not last_batch else size
        yield X[idx:end],Y[idx:end]
        idx = end if not last_batch else 0
   
    