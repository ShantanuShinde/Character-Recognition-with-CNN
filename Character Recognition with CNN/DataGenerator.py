from keras.utils import Sequence
import os
from keras.utils import HDF5Matrix
from cv2 import imread
import numpy as np

class DataGenerator(Sequence):
    def __init__(self,dataFile,type_data,batch_size,shuffle=True):
        self.dataFile = dataFile
        self.type_data = type_data
        self.y = np.array(HDF5Matrix(self.dataFile,"/" + self.type_data+"_Y/Ydata"))
        
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        self.indices = np.arange(len(self.y))
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __data__generation(self,indices):
        X = np.empty((self.batch_size,278,278,1))
        Y = np.empty((self.batch_size,278*278,1),dtype=int)

        for i,ind in enumerate(indices):
            X[i] = np.array(HDF5Matrix(self.dataFile,"/" + self.type_data + "_X/X" + str(ind)))
            Y[i] = np.reshape(np.array(HDF5Matrix(self.dataFile,"/label/labels" + str(self.y[i]))),(278*278,1))


        return X,Y        

    def __len__(self):
        return int(np.ceil(len(self.y) / float(self.batch_size)))

    def __getitem__(self,index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        X,Y = self.__data__generation(indices)
        #print(X.shape,Y.shape)
        return X,Y
    



            
