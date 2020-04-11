import numpy as np
import cv2

def rgb2gray(img):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    res = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return res.astype(np.uint8)
def maxmin(img):
    k_size = 3
    pad = k_size//2
    h,w= img.shape
    res = np.zeros((h+pad*2, w+pad*2))
    res[1:h+pad,1:w+pad] = img[...]
    temp = res.copy()
    for i in range(1,h+pad+1):
        for j in range(1,w+pad+1):
            res[i,j] = np.max(temp[i-pad:i+pad+1,j-pad:j+pad+1])-np.min(temp[i-pad:i+pad+1,j-pad:j+pad+1])
    res = res[1:h+pad,1:w+pad]

    return res.astype(np.uint8)


def main():
    img = cv2.imread("imori.jpg")
    img = rgb2gray(img)
    res = maxmin(img)
    cv2.imwrite("result13.jpg",res)
    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()