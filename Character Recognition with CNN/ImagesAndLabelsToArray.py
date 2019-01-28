import cv2
import numpy as np
import os
import h5py
from tables import *
from keras.utils import to_categorical


path="A:\Code\Machine Learning\Character Recognition\Processed Training Dataset"

data = open_file("dataset.h5",mode="w")

labelDataGroup = data.create_group(data.root,"label")

for i in range(27):
     
    lbls = data.create_array(labelDataGroup,"labels"+str(i),obj=im)


XdataGroup = data.create_group(data.root,"train_X")



#Save images into h5 file using Pytables
#i=0
#files = list(enumerate(os.scandir(path)))
#for m,folders in files:
#   folder = list(enumerate(os.scandir(folders.path)))
#   for n,img in folder:
      
#       im = cv2.imread(img.path,0)
#       im = im/255
#       im = np.expand_dims(im,2)
#       im = np.expand_dims(im,0)
#       if not i:
#           X = data.create_earray(dataGroup,"X",obj=im)
#       else:
#           X.append(im)
#       i=1
#       print(img)
       
files = list(enumerate(os.scandir(path)))
for i,img in files:
    im = np.expand_dims(cv2.imread(img.path,0),2)
    print(img)
    im = im/255
    dataIm = data.create_array(XdataGroup,"X"+str(i),obj=im)
    
        
print("Training images saved into array")


#Conver labels to one hot vectors and save into the h5 file
Y_1 = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append([0 for i in range(54)],[1 for i in range(1211)]),[2 for i in range(649)]),[3 for i in range(786)]),[4 for i in range(698)]),[5 for i in range(1172)]),[6 for i in range(598)]),[7 for i in range(688)]),[8 for i in range(754)]),[9 for i in range(905)]),[10 for i in range(568)]),[11 for i in range(586)]),[12 for i in range(796)]),[13 for i in range(739)]),[14 for i in range(1019)]),[15 for i in range(995)]),[16 for i in range(722)]),[17 for i in range(573)]),[18 for i in range(985)]),[19 for i in range(985)]),[20 for i in range(900)]),[21 for i in range(608)]),[22 for i in range(597)]),[23 for i in range(568)]),[24 for i in range(605)]),[25 for i in range(655)]),[26 for i in range(573)])
#Y_1 = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(["None" for i in range(20)],["A" for i in range(20)]),["B" for i in range(20)]),["C" for i in range(20)]),["D" for i in range(20)]),["E" for i in range(20)]),["F" for i in range(20)]),["G" for i in range(20)]),["H" for i in range(20)]),["I" for i in range(20)]),["J" for i in range(20)]),["K" for i in range(20)]),["L" for i in range(20)]),["M" for i in range(20)]),["N" for i in range(20)]),["O" for i in range(20)]),["P" for i in range(20)]),["Q" for i in range(20)]),["R" for i in range(20)]),["S" for i in range(20)]),["T" for i in range(20)]),["U" for i in range(20)]),["V" for i in range(20)]),["W" for i in range(20)]),["X" for i in range(20)]),["Y" for i in range(20)]),["Z" for i in range(20)])


#alpha = [x for x in au]

#alpha.insert(0,"None")

#dic = {let:i for i,let in enumerate(alpha)}

#Y_1int = [dic[char] for char in Y_1]

#Y_1OneHot = to_categorical(Y_1int,dtype='int32')

YdataGroup = data.create_group(data.root,"Y")

Y = data.create_earray(YdataGroup,"Ydata",obj=Y_1)



print("Training labels saved in array")


#print(" Label shape: " + str(YdataGroup["Ydata"].shape))




#Save testing data

path2 = "A:\Code\Machine Learning\Character Recognition\Processed Testing Dataset"


XtestDataGroup = data.create_group(data.root,"test_X")

files = list(enumerate(os.scandir(path2)))
for i,img in files:
    im = np.expand_dims(cv2.imread(img.path,0),2)
    im = im/255
    dataIm = data.create_array(XtestDataGroup,"X" +str(i),obj=im)
    print(img)

#i=0
#files = list(enumerate(os.scandir(path2)))
#for m,folders in files:
#   folder = list(enumerate(os.scandir(folders.path)))
#   for n,img in folder:

#       im = cv2.imread(img.path,0)
#       im = im/255 
#       im = np.expand_dims(im,2)
#       im = np.expand_dims(im,0)
#       if not i:
#           X = testData.create_earray(testDataGroup,"X",obj=im)
#       else:
#           X.append(im)
#       i=1
#       print(img)



print("Test images saved in array")

Y_1 = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append([1 for i in range(512)],[2 for i in range(76)]),[3 for i in range(164)]),[4 for i in range(198)]),[5 for i in range(432)]),[6 for i in range(74)]),[7 for i in range(104)]),[8 for i in range(147)]),[9 for i in range(365)]),[10 for i in range(31)]),[11 for i in range(59)]),[12 for i in range(208)]),[13 for i in range(144)]),[14 for i in range(306)]),[15 for i in range(310)]),[16 for i in range(138)]),[17 for i in range(8)]),[19 for i in range(346)]),[19 for i in range(263)]),[20 for i in range(301)]),[21 for i in range(110)]),[22 for i in range(55)]),[23 for i in range(56)]),[24 for i in range(15)]),[25 for i in range(40)]),[26 for i in range(8)])
#Y_1 = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(["A" for i in range(10)],["B" for i in range(10)]),["C" for i in range(10)]),["D" for i in range(10)]),["E" for i in range(10)]),["F" for i in range(10)]),["G" for i in range(10)]),["H" for i in range(10)]),["I" for i in range(10)]),["J" for i in range(10)]),["K" for i in range(10)]),["L" for i in range(10)]),["M" for i in range(10)]),["N" for i in range(10)]),["O" for i in range(10)]),["P" for i in range(10)]),["Q" for i in range(8)]),["R" for i in range(10)]),["S" for i in range(10)]),["T" for i in range(10)]),["U" for i in range(10)]),["V" for i in range(10)]),["W" for i in range(10)]),["X" for i in range(10)]),["Y" for i in range(10)]),["Z" for i in range(8)])

#alpha = [x for x in au]

#alpha.insert(0,"None")

#dic = {let:i for i,let in enumerate(alpha)}

#Y_1int = [dic[char] for char in Y_1]

#Y_1OneHot = to_categorical(Y_1int,dtype='int32')

YtestDataGroup = data.create_group(data.root,"test_Y")

Y = data.create_array(YtestDataGroup,"Ydata",obj = Y_1)



print("Test labels saved in array")

#print(" Label shape: " + str(YtestDataGroup["Ydata"].shape))


data.flush()

data.close()
