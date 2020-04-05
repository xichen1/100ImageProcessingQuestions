import cv2
import numpy as np
def grayscale(img):
    b = img[:,:,0]
    g = img[:,:,1]
    r = img[:,:,2]
    result = 0.2126*r + 0.7152*g + 0.0722*b
    result = result.astype(np.uint8) # convert the float to uint8 because cv2 cannot imwrite float img
    return result

def main():
	img = cv2.imread("imori.jpg")
	out = grayscale(img)
	cv2.imwrite("out.jpg", out)
	cv2.imshow("result", out)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

main()