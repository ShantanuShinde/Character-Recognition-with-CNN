import cv2
import numpy as np

# create MSER object
mser = cv2.MSER_create()

# read the image
img = cv2.imread('textArea01.png')

# convert to gray scale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# store copy of the image
vis = img.copy()

# detect regions in the image
regions,_ = mser.detectRegions(gray)

# find convex hulls of the regions and draw them onto the original image
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

cv2.polylines(vis, hulls, 1, (0, 255, 0))

# create mask for the detected region
mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
mask = cv2.dilate(mask, np.ones((150, 150), np.uint8))

# initialize threshold area of the contours
ThresholdContourArea = 10000
for contour in hulls:
    # use the contour on if area less than threshold
    if cv2.contourArea(contour) > ThresholdContourArea:
        continue
    cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
    #this is used to find only text regions, remaining are ignored
    text_only = cv2.bitwise_and(img, img, mask=mask)

# display the countours, mask and the final image and save them
cv2.imshow('img', vis)
cv2.waitKey(0)
cv2.imwrite('contours.png',vis)
cv2.imshow('mask', mask)
cv2.waitKey(0)
cv2.imwrite('mask.png',mask)
cv2.imshow('text', text_only)
cv2.waitKey(0)
cv2.imwrite('final.png',text_only)