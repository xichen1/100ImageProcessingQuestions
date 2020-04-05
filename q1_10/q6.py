import cv2
import numpy as np

def reduce(img):
	reduced = np.zeros_like(img)
	b = img[...,0].copy()
	g = img[...,1].copy()
	r = img[...,2].copy()
	values = [32,96,160,224]
	for i in range(0,255,64):
		idx = np.where((b>=i) & (b<=i+64)) 
		reduced[...,0][idx] = values[int(i/64)]

		idx = np.where((g>=i) & (g<=i+64)) 
		reduced[...,1][idx] = values[int(i/64)]

		idx = np.where((r>=i) & (r<=i+64)) 
		reduced[...,2][idx] = values[int(i/64)]
	print(reduced[0])
	return reduced

# observe the values 32 = 0*64 + 32, 96 = 1*64 + 32, 160 = 2*64 + 32
def new_reduce(img):
	out = img.copy()
	out = out//64 * 64 + 32
	return out

img = cv2.imread("imori.jpg")
out = new_reduce(img)
cv2.imwrite("result6.jpg", out)
cv2.imshow("",out)
cv2.waitKey(0)
cv2.destroyAllWindows()
