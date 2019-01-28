import cv2
import numpy as np
import PredictImage

imagePath = "A:\Code\Machine Learning\Character Recognition\English\Img\BadImag\Bmp\Sample048\img048-00022.png"

img = cv2.imread(imagePath)
img = cv2.resize(img,(60,90))
img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
X = np.array([img])
Y = PredictImage.Predict(X)
print(Y)