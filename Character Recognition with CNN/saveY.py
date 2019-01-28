import numpy as np
from tables import *
from string import ascii_uppercase as au
from keras.utils import to_categorical

data = open_file("small_Y_train_dataset.h5",mode="w")

dataGroup = data.create_group(data.root,"Data")

Y_1 = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(["None" for i in range(20)],["A" for i in range(20)]),["B" for i in range(20)]),["C" for i in range(20)]),["D" for i in range(20)]),["E" for i in range(20)]),["F" for i in range(20)]),["G" for i in range(20)]),["H" for i in range(20)]),["I" for i in range(20)]),["J" for i in range(20)]),["K" for i in range(20)]),["L" for i in range(20)]),["M" for i in range(20)]),["N" for i in range(20)]),["O" for i in range(20)]),["P" for i in range(20)]),["Q" for i in range(20)]),["R" for i in range(20)]),["S" for i in range(20)]),["T" for i in range(20)]),["U" for i in range(20)]),["V" for i in range(20)]),["W" for i in range(20)]),["X" for i in range(20)]),["Y" for i in range(20)]),["Z" for i in range(20)])


alpha = [x for x in au]

alpha.insert(0,"None")

dic = {let:i for i,let in enumerate(alpha)}

Y_1int = [dic[char] for char in Y_1]

Y_1OneHot = to_categorical(Y_1int,dtype='int32')

Y = data.create_earray(dataGroup,"Y",obj=Y_1OneHot)

data.flush()

data.close()


testData = open_file("small_Y_test_dataset.h5",mode="w")

testDataGroup = testData.create_group(testData.root,"Data")

Y_1 = np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(np.append(["A" for i in range(10)],["B" for i in range(10)]),["C" for i in range(10)]),["D" for i in range(10)]),["E" for i in range(10)]),["F" for i in range(10)]),["G" for i in range(10)]),["H" for i in range(10)]),["I" for i in range(10)]),["J" for i in range(10)]),["K" for i in range(10)]),["L" for i in range(10)]),["M" for i in range(10)]),["N" for i in range(10)]),["O" for i in range(10)]),["P" for i in range(10)]),["Q" for i in range(8)]),["R" for i in range(10)]),["S" for i in range(10)]),["T" for i in range(10)]),["U" for i in range(10)]),["V" for i in range(10)]),["W" for i in range(10)]),["X" for i in range(10)]),["Y" for i in range(10)]),["Z" for i in range(8)])

alpha = [x for x in au]

alpha.insert(0,"None")

dic = {let:i for i,let in enumerate(alpha)}

Y_1int = [dic[char] for char in Y_1]

Y_1OneHot = to_categorical(Y_1int,dtype='int32')

Y = testData.create_earray(testDataGroup,"Y",obj=Y_1OneHot)

testData.flush()

testData.close()

