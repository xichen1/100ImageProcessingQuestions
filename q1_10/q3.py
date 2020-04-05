import cv2
import numpy as np

def thresholding(img):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    result = 0.2126*r + 0.7152*g + 0.0722*b
    result = result.astype(np.uint8)
    return result

img = cv2.imread("imori.jpg")
result = thresholding(img)
result[result>128] = 255
result[result<=128] = 0
cv2.imwrite("result3.jpg", result)
cv2.imshow("result", result)
cv2.waitKey(0)
cv2.destroyAllWindows()

