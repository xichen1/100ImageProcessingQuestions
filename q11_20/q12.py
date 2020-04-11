import numpy as np
import cv2


def diag(img):
    k_size = 3
    pad = k_size//2
    h,w,c= img.shape
    res = np.zeros((h+pad*2, w+pad*2, c),dtype=np.float)
    res[1:h+pad,1:w+pad,:] = img[...]
    temp = res.copy()
    for i in range(1,h+pad):
        for j in range(1,w+pad):
            for z in range(c):
                avg = (temp[i-1,j-1,z]+temp[i,j,z]+temp[i+1,j+1,z])/3
                res[i,j,z] = avg
    res = res[1:h+pad,1:w+pad,:]

    return res.astype(np.uint8)


def main():
    img = cv2.imread("imori.jpg")
    res = diag(img)
    cv2.imwrite("result12.jpg",res)
    cv2.imshow("res", res)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

main()