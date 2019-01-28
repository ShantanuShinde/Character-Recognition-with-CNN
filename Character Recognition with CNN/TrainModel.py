import LoadData
import Model
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'



Model.TrainAndSave()