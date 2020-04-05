import cv2
import numpy as np

def gaussian(img):
    h,w,c = img.shape
    k_size = 3
    # pad
    pad = k_size//2
    # imshow can only accept uint8, double or float32
    out = np.zeros((h+pad*2, w+pad*2, c),dtype=np.uint8)
    out[1:1+h,1:1+w,:] = img[...].copy()



    return out

img = cv2.imread("imori_noise.jpg")
out = gaussian(img)
cv2.imshow("",out)
cv2.waitKey(0)
cv2.destroyAllWindows()