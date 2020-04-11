import numpy as np
import cv2


def rgb2gray(img):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    res = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return res.astype(np.uint8)
def lap(img):
    h,w = img.shape
    k_size = 3
    pad = k_size//2
    res = np.zeros((h+2*pad, w+2*pad),dtype=np.float)
    res[pad:pad+h,pad:pad+w] = img
    temp = res.copy()
    for i in range(pad,h+pad):
        for j in range(pad,w+pad):
            res[i,j] = temp[i-1][j]+temp[i][j-1]-4*temp[i][j]+temp[i+1][j]+temp[i][j+1]
    res = np.clip(res,0,255)
    return res.astype(np.uint8)

def main():
    img = cv2.imread("imori.jpg")
    img = rgb2gray(img)
    res = lap(img)
    cv2.imwrite("result17.jpg",res)
    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()