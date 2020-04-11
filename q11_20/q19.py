# Write a Laplacian of Gaussian filter
import numpy as np
import cv2

def rgb2gray(img):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    res = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return res.astype(np.uint8)

def LoG(img):
    h,w = img.shape
    sigma=3
    k_size = 5
    pad = k_size//2
    res = np.zeros((h+pad*2,w+pad*2))
    res[pad:pad+h,pad:pad+w] = img
    k = np.zeros((k_size,k_size))
    for i in range(k_size):
        for j in range(k_size):
            k[i,j] = ((i-pad)**2+(j-pad)**2-sigma**2)*np.exp(-((i-pad)**2+(j-pad)**2)/(2*(sigma**2)))
    k /= (2*np.pi*(sigma**6))
    k /= np.sum(k)

    temp = res.copy()
    for i in range(pad,h+pad):
        for j in range(pad,pad+w):
            res[i,j] = np.sum(k*temp[i-pad:i+pad+1, j-pad:j+pad+1])
    res = res[pad:h+pad,pad:pad+w].astype(np.uint8)
    return res

def main():
    img = cv2.imread("imori_noise.jpg")
    img = rgb2gray(img)
    res = LoG(img)
    cv2.imwrite("result19.jpg",res)
    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()