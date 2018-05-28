import cv2
import numpy as np

img = cv2.imread('Ex.png')
img_bw = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# cv2.imshow('bw', img_bw)
# cv2.waitKey(0)
se1 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
se2 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
mask = cv2.morphologyEx(img_bw, cv2.MORPH_TOPHAT, se1)
mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, se2)
mask = np.dstack([mask, mask, mask])/255
out = img * mask
cv2.imshow('Output', out)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('output.png', out)