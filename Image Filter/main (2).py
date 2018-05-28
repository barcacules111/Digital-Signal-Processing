import cv2
import numpy as np
from scipy import interpolate

image = cv2.imread('T1.png')
gray_im = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# new_gray_image = np.zeros((gray_im.shape[0]+60, gray_im.shape[1]+60))
# new_gray_image[30:-30, 30:-30] = gray_im
# gray_im = new_gray_image
cv2.imshow('im', gray_im)
cv2.waitKey(0)
n = 46
# grad = 40
B0 = np.zeros([1,1], np.uint8)
Bn = np.zeros([n,n], np.uint8)

for i in range(n//2):
    Bn[i:n-i, i:n-i] = n*(n//2-i)
Bn[n//2,n//2]=0
print(Bn)
# Bn[1:n-1,1:n-1] = 255
erosion = lambda img, kernel: cv2.erode(img, kernel, iterations=1)
dilation = lambda img, kernel: cv2.dilate(img, kernel, iterations=1)

# c =  erosion(erosion(dilation(dilation(gray_im, B0), Bn), Bn), B0)
c = dilation(gray_im, B0)
# cv2.imshow('dil1',c)
# cv2.waitKey(0)
c = dilation(c, Bn)
# c = cv2.resize(c, (gray_im.shape[1], gray_im.shape[0]))
# cv2.imshow('dil2',c)
# cv2.waitKey(0)
c = erosion(c, Bn)
# c = cv2.resize(c, (gray_im.shape[1], gray_im.shape[0]))
# cv2.imshow('eros3',c)
# cv2.waitKey(0)
c = erosion(c, B0)
# cv2.imshow('eros4',c)
# cv2.waitKey(0)
c -= gray_im
# cv2.imshow('-gray',c)
# cv2.waitKey(0)
c[c<200] = 0
cv2.imshow('<250',c)
cv2.waitKey(0)
c = dilation(c, np.ones((3,3),np.uint8))
c = erosion(c, np.ones((4,4),np.uint8))
c = dilation(c, np.ones((4,4),np.uint8))
# c *= (0.3 * gray_im.astype(np.float64)).astype(np.uint8)
# c = gray_im - np.round(c)
cv2.imshow('dilation',c)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
wind_size = 20
for x in range(0,gray_im.shape[0]-wind_size,3):
    for y in range(0,gray_im.shape[1]-wind_size,3):
        non_zx, non_zy = np.where(c[x:x + wind_size, y:y + wind_size] > 0)
        if len(non_zx)>0:
            non_zx += x
            non_zy += y
            ind1, ind2 = np.where(c[x:x + wind_size, y:y + wind_size] == 0)
            ind1 += x
            ind2 += y
            # print(ind1+x, ind2+y)
            # print(gray_im[ind1+x,ind2+y])
            f = interpolate.interp2d(ind1, ind2, gray_im[ind1,ind2], kind='cubic')
            res = np.round(f(non_zx, non_zy)).astype(np.uint8)
            # print(res)
            if len(res.shape) == 1:
                gray_im[non_zx, non_zy] = res
            elif len(res.shape) == 2:
                gray_im[non_zx, non_zy] = res[:, 0]
            else:
                print(res, non_zy, non_zx)
            # try:
            #     gray_im[non_zx, non_zy] = res
            # except:
            #     print('i1 ',ind1)
            #     print('i2', ind2)
            #     print(non_zx)
            #     print(non_zy)
            #     print(res)
            #     quit(0)
            # print(gray_im[non_zx, non_zy])
            # quit(0)
        # if c[x:x+wind_size,y:y+wind_size].any()>0:
        #     print(c[x:x+wind_size,y:y+wind_size])
        #     print(np.where(c[x:x+wind_size,y:y+wind_size]>0))'''
# cv2.imshow('res', gray_im)
# cv2.waitKey(0)
# cv2.destroyAllWindows()