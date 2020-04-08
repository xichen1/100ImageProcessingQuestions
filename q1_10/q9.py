import cv2
import numpy as np

def gaussian(img):
    h,w,c = img.shape
    k_size = 3
    # pad
    pad = k_size//2
    # imshow can only accept uint8, double or float32
    k = np.zeros((k_size,k_size))
    out = np.zeros((h+pad*2, w+pad*2, c),dtype = np.float)
    out[1:1+h,1:1+w,:] = img[...].copy()
    for i in range(-pad, k_size-pad):
        for j in range(-pad, k_size-pad):
            k[i+pad][j+pad] = np.exp(-(i**2+j**2)/2*1.1**2)/(2*np.pi*1.**2)
    k = k/np.sum(k)
    # aaa = out[1-k_size:1+k_size][1-k_size:1+k_size][0]

    # print(aaa)
    for z in range(c):
        for i in range (1,h+1):
            for j in range(1,w+1):
                out[i,j,z] = np.sum(out[i-pad:i+pad+1,j-pad:j+pad+1,z]*k)
    out = np.clip(out, 0, 255)
    out = out.astype(np.uint8)

    return out

img = cv2.imread("imori_noise.jpg")
out = gaussian(img)
cv2.imwrite("result9.jpg",out)
cv2.imshow("",out)
cv2.waitKey(0)
cv2.destroyAllWindows()