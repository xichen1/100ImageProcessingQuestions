import numpy as np
import cv2


def rgb2gray(img):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    res = 0.2126 * r + 0.7152 * g + 0.0722 * b
    return res.astype(np.uint8)

def differ(img):
    vm = np.array([[0,-1,0],[0,1,0],[0,0,0]])
    hm = np.array([[0,0,0],[-1,1,0],[0,0,0]])
    k_size =3
    pad = k_size//2
    h,w = img.shape
    hd = np.zeros((h+pad*2,w+pad*2)).astype(np.float)
    hd[1:h+pad,1:w+pad] = img
    vd = hd.copy()
    temp = hd.copy()

    for i in range(1,h+pad):
        for j in range(1,w+pad):
            hd[i,j] = np.sum(hm*temp[i-pad:i+pad+1, j-pad:j+pad+1])
            vd[i,j] = np.sum(vm*temp[i-pad:i+pad+1, j-pad:j+pad+1])
    hd = np.clip(hd, 0, 255)
    vd = np.clip(vd, 0, 255)
    hd = hd[pad:h+pad,pad:w+pad].astype(np.uint8)
    vd = vd[pad:h+pad,pad:w+pad].astype(np.uint8)
    return hd,vd


def main():
    img = cv2.imread("imori.jpg")
    img = rgb2gray(img)
    hd,vd = differ(img)
    cv2.imwrite("result14_v.jpg",vd)
    cv2.imwrite("result14_h.jpg",hd)
    cv2.imshow("hor", hd)
    cv2.imshow("ver", vd)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
main()