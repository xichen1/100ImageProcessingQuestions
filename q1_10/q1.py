import cv2
def rgb2bgr(img):

	red = img[:,:,2].copy()
	green = img[:,:,1].copy()
	blue = img[:,:,0].copy()

	img[:,:,0] = red
	img[:,:,1] = green
	img[:,:,2] = blue
	return img
img_ = cv2.imread("imori.jpg")
res = rgb2bgr(img_)
cv2.imwrite("result1.jpg",res)
cv2.imshow("result", res)
cv2.waitKey(0)
cv2.destroyAllWindows()