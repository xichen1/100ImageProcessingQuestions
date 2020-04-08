import numpy as np
import cv2

def median(img):
    k_size = 3
    pad = k_size//2
    h,w,c = img.shape
    out = np.zeros((h+pad*2,w+pad*2,c),dtype=np.float)
    out[1:h+1,1:w+1,:] = img

    for i in range(1,h+1):
        for j in range(1,w+1):
            for z in range(3):
                out[i,j,z] = np.median(out[i-pad:i+pad+1,j-pad:j+pad+1,z])

    return out.astype(np.uint8)
img = cv2.imread("imori_noise.jpg")
out = median(img)
cv2.imwrite("result10.jpg",out)
cv2.imshow("",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
