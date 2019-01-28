from keras.layers import *
from keras.models import Sequential,Model
from keras.optimizers import *
import matplotlib.pyplot as plt
from LoadData import Generator
import numpy as np
from keras.regularizers import *
from keras.utils import plot_model
import os
from DataGenerator import DataGenerator

os.environ["PATH"] += os.pathsep + 'C:/Program Files (x86)/Graphviz2.38/bin/'

def CompileModel():
    #model = Sequential()
    inp = Input((278,278,1))

    #Encoding Layer

    #1st Block
    conv1 = Conv2D(16,(3,3),padding="same")(inp)
    batch1 = BatchNormalization()(conv1)
    act1 = Activation("relu")(batch1)
    maxpool1 = MaxPool2D()(act1)
    print(maxpool1.shape)

    #2nd A Block
    conv2A = Conv2D(8,(2,2),padding="same")(maxpool1)
    act2A = Activation("relu")(conv2A)
    #maxpool2A = MaxPool2D(padding="same")(conv2A)
    print(conv2A.shape)

    #2nd B Block
    conv2B = Conv2D(8,(2,2),padding="same")(maxpool1)
    act2B = Activation("relu")(conv2B)
    #maxpool2B = MaxPool2D(padding="same")(conv2B)
    print(conv2B.shape)

    #Decoding Layer

    #3rd A Block
    
    conv3A = Conv2D(8,(2,2),padding="same")(act2A)
    act3A = Activation("relu")(conv3A)
    print(conv3A.shape)

    

    #Concatenate
    concat = Concatenate(3)([act2B,act3A])
    print(concat.shape)

    #4th Block
    upsamp2 = UpSampling2D()(concat)
    conv4 = Conv2D(16,(3,3),padding="same")(upsamp2)
    batch2 = BatchNormalization()(conv4)
    act4 = Activation("relu")(batch2)
    print(batch2.shape)

    conv5 = Conv2D(1,(3,3),padding="same")(act4)
    print(conv5.shape)
    
    #Permute
    reshp = Reshape((1,77284))(conv5)
    
    permt = Permute((2,1))(reshp)
    act5 = Activation("softmax")(permt)

    model = Model(input=inp,output=act5)

    #Compile model
    adm = Adagrad(lr=0.02)
    model.compile(optimizer=adm,loss="categorical_crossentropy",metrics=['accuracy'])
    print("Model Compiled")
    return model

def TrainAndSave(batch_size=20,epochs=100):

    #train_generator = Generator("small_dataset.h5",batch_size)
    #test_generator = Generator("small_test_dataset.h5",batch_size)
    #X_train,Y_train = Generator("small_dataset.h5",batch_size)
    #X_test,Y_test = Generator("small_test_dataset.h5",batch_size)
    train_generator = DataGenerator("dataset.h5","train",batch_size)
    test_generator = DataGenerator("dataset.h5","test",batch_size)
    model = CompileModel()
    model.summary()
    plot_model(model,to_file="model.jpeg",show_shapes=True,show_layer_names=True,rankdir="TB")
    #history = model.fit(x = X_train,y=Y_train,batch_size=batch_size,epochs=epochs,validation_data=(X_test,Y_test),shuffle='batch')
    history = model.fit_generator(generator = train_generator,epochs = epochs,validation_data=test_generator,use_multiprocessing=False)
   
    
    # Plot training & validation accuracy values
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()

    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')

    plt.legend(['Train'], loc='upper left')
    plt.show()
    plt.savefig("Training graph.jpeg")
    model.save("model.h5")


if __name__ == "__main__":
    TrainAndSave()




