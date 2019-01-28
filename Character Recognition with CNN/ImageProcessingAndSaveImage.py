import cv2
import numpy as np
import os
from string import ascii_uppercase

path1="A:/Code/Machine Learning/Character Recognition/UnProcessed Training Dataset/New folder"
path2 = "A:/Code/Machine Learning/Character Recognition/UnProcessed Training Dataset/Latin"
dest = "A:/Code/Machine Learning/Character Recognition/Processed Training Dataset"



def IncreaseImageContrast(img):
    #Convert image to LAB mode
    lab = cv2.cvtColor(img,cv2.COLOR_BGR2LAB )
    #split it into l,a,b, channels
    l,a,b = cv2.split(lab)
    #Apply clahe to the l channel
    clahe = cv2.createCLAHE(clipLimit=3.0,tileGridSize=(8,8))
    cl = clahe.apply(l)
    #Merge layers
    limg = cv2.merge((cl,a,b))
    #Convert LAB to BGR
    fImg = cv2.cvtColor(limg,cv2.COLOR_LAB2BGR)
    return fImg



#Process and positive save samples

i = 0
l = len(ascii_uppercase)
files = list(enumerate(os.scandir(path1)))
for m,folders in files:
   folder = list(enumerate(os.scandir(folders.path)))
   for n,img in folder:
        im = cv2.imread(img.path)
        print(img)
        im = cv2.resize(im,(278,278))
        im = IncreaseImageContrast(im)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)  #Convert image to gray
        im = cv2.fastNlMeansDenoising(im,None,10,7,21)  #Remove noise
        thresh,im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)    #Convert to binary(black and white)
        thresh,im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite(dest + "/" + ascii_uppercase[i%l] + "/" +  str(n+1000*i) + ".jpg" , im)
   i+=1

i = 0
l = len(ascii_uppercase)
files = list(enumerate(os.scandir(path2)))
for m,folders in files:
   folder = list(enumerate(os.scandir(folders.path)))
   for n,img in folder:
        im = cv2.imread(img.path,-1)
        print(img)
        for x in range(im.shape[0]):
            for y in range(im.shape[1]):
                if im[x,y][3] < 255:
                    im[x,y] = (255,255,255,255)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        cv2.imwrite(dest + "/" + ascii_uppercase[i%l] + "/" +  str(n+10000*(i+1)) + ".jpg" , im)
   i+=1


#Process and save negative samples

folder = list(enumerate(os.scandir(path1+"/NegativeSamples")))
for n,img in folder:
    im = cv2.imread(img.path)
    print(img)
    im = cv2.resize(im,(278,278))
    im = IncreaseImageContrast(im)
    im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)  #Convert image to gray
    im = cv2.fastNlMeansDenoising(im,None,10,7,21)  #Remove noise
    thresh,im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)    #Convert to binary(black and white)
    thresh,im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite(dest + "/Negative Samples" + str(n) + ".jpg",im)


testPath = "A:/Code/Machine Learning/Character Recognition/UnProcessed Testing Dataset"
destPath = "A:/Code/Machine Learning/Character Recognition/Processed Testing Dataset"

#Process and save test samples

i = 0
files = list(enumerate(os.scandir(testPath)))
for m,folders in files:
   folder = list(enumerate(os.scandir(folders.path)))
   for n,img in folder:
        im = cv2.imread(img.path)
        print(img)
        im = cv2.resize(im,(278,278))
        im = IncreaseImageContrast(im)
        im = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)  #Convert image to gray
        im = cv2.fastNlMeansDenoising(im,None,10,7,21)  #Remove noise
        thresh,im = cv2.threshold(im, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)    #Convert to binary(black and white)
        thresh,im = cv2.threshold(im, thresh, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        cv2.imwrite(destPath + "/" + ascii_uppercase[i] + "/" +  str(n) + ".jpg" , im)
   i+=1



