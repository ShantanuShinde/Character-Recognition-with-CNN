import cv2
import imutils

img = cv2.imread('textArea01.png')
img = imutils.resize(img, width=500)

# remove border
kern_ver = cv2.getStructuringElement(cv2.MORPH_RECT,(1,50))
temp1 = 255 - cv2.morphologyEx(img, cv2.MORPH_CLOSE, kern_ver)
kern_hor = cv2.getStructuringElement(cv2.MORPH_RECT, (50,1))
temp2 = 255 - cv2.morphologyEx(img, cv2.MORPH_CLOSE, kern_hor)
temp3 = cv2.add(temp1,temp2)
result = cv2.add(temp3, img)

# covert to grayscale and threshold
gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY)
thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

# find contours and filter using contour area
cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
cnts = cnts[0] if len(cnts) == 2 else cnts[1]
for c in cnts:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(result, (x,y), (x+w, y+h), (36, 255, 12), 2)

cv2.imshow("thresh", thresh)
cv2.imshow("result", result)
cv2.waitKey()

