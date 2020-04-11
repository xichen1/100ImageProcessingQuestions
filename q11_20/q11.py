import numpy as np
import cv2


def meanfilter(img):
    k_size = 3
    pad = k_size//2
    h,w,c= img.shape
    res = np.zeros((h+pad*2, w+pad*2, c))
    res[1:h+pad,1:w+pad,:] = img[...]
    temp = res.copy()
    for i in range(1,h+pad+1):
        for j in range(1,w+pad+1):
            for z in range(c):
                res[i,j,z] = np.mean(temp[i-pad:i+pad+1,j-pad:j+pad+1,z])
    res = res[1:h+pad,1:w+pad,:]

    return res.astype(np.uint8)


def main():
    img = cv2.imread("imori.jpg")
    res = meanfilter(img)
    cv2.imwrite("result11.jpg",res)
    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()