import cv2
import numpy as np

def avgpool(img):
	h, w, c = img.shape
	out = img.copy()
	for i in range(c):
		for y in range(0,h,8):
			for x in range(0,w,8):
				out[y:y+8,x:x+8,i] = (np.mean(img[y:y+8,x:x+8,i])).astype(np.int)
				# or (np.mean(img[y:y+8,x:x+8,i]))/64
	return out
	

img = cv2.imread("imori.jpg")
out = avgpool(img)
cv2.imwrite("result7.jpg", out)
cv2.imshow("", out)
cv2.waitKey(0)
cv2.destroyAllWindows()