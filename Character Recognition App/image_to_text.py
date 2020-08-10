from keras.models import load_model
import numpy as np
import cv2
from string import ascii_uppercase as UC
import matplotlib.pyplot as plt
from keras.utils import to_categorical

MODEL_PATH = "E:\Code\Machine Learning\Character-Recognition-with-CNN\Using Modified LeNet-5 Model\trained_model.h5"

def get_char_rect(img):
  """
  This function takes in an images and returns bounding rectangles for characters in the image
  Parameters : img (Image)
  Return : List containing top left location of the bounding rectangle and it's length and breadth
  """
  

  # convert to grayscale and threshold
  gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
  thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

  # find contours and filter out using area
  cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
  cnts = cnts[0] if len(cnts) == 2 else cnts[1]

  rects = []
  for c in cnts:
      x, y, w, h = cv2.boundingRect(c)
      rects.append([x,y,w,h])
  
  rects = np.array(rects)
  max_height = np.max(rects[::, 3])
  nearest = max_height * 1.4
  
  # sort rectangles
  sorted_rects = sorted(rects, key=lambda rect:   [int(nearest * round(float(rect[1]) / nearest)), rect[0]])
  
  return sorted_rects

def get_text(img):
    '''
    This function takes in an RGB image in form of numpy array and returns a list of letters in the image
    '''
    model = load_model(MODEL_PATH)

    rects = get_char_rect(img)
    letter_list = list(UC)

    letters = []
    imgs = []
    for rect in rects:
        x, y, w, h = rect
        im = img[y:y+h, x:x+w]

        im = cv2.resize(im,(24,24))
        im = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        im = np.pad(im,(2,2), constant_values=255)

        

        im = np.expand_dims(im, axis=2)
        imgs.append(im)
        ims = np.array([im])

        pred = model.predict(ims)
        pred = np.argmax(pred,axis=1)

        letters.append(letter_list[pred[0]])
    
    return letters, imgs

def train_model(imgs,lbls):
    imgs = np.array(imgs)
    lbls = lbls

    one_ht = to_categorical(range(26))

    oh_lbls = []
    for l in lbls:
        oh_lbls.append(one_ht[l])

    
    oh_lbls = np.array(oh_lbls)
    
    model = load_model(MODEL_PATH)

    model.fit(x=imgs,y=oh_lbls)

    model.save(MODEL_PATH)