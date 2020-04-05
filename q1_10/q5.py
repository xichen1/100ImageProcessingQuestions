import cv2
import numpy as np


def bgr2hsv(_img):

	img = _img.copy()/255

	# for every pixel, get the largest value from r, g, b
	# axis = 2 means compare and get value from img[..][..][..]
	max_v = np.max(img, axis=2).copy() 
	min_v = np.min(img, axis=2).copy()

	# use argmin to find which channel the min value are from
	min_idx = np.argmin(img, axis=2)

	hsv = np.zeros_like(img,dtype=np.float32)
	hsv[:,:,0][np.where(max_v==min_v)] = 0

	min_b = np.where(min_idx==0)
	hsv[:,:,0][min_b] = 60*(img[:,:,1][min_b] - img[:,:,2][min_b])/(max_v[min_b]-min_v[min_b])+60

	min_g = np.where(min_idx==1)
	hsv[:,:,0][min_g] = 60*(img[:,:,2][min_g] - img[:,:,0][min_g])/(max_v[min_g]-min_v[min_g])+300

	min_r = np.where(min_idx==2)
	hsv[:,:,0][min_r] = 60*(img[:,:,0][min_r] - img[:,:,1][min_r])/(max_v[min_r]-min_v[min_r])+180

	hsv[:,:,1] = max_v.copy()-min_v.copy()

	hsv[:,:,2] = max_v.copy()

	return hsv

def hsv2bgr(_img, hsv):
	img = _img.copy()/255

	bgr = np.zeros_like(img)
	h = hsv[...,0]
	s = hsv[...,1]
	v = hsv[...,2]
	h_1 = h/60
	c = s
	x = c*(1-np.abs(h_1%2 - 1))
	zero = np.zeros_like(h)
	b1g1r1 = [[zero,x,c],[zero,c,x],[x,c,zero],[c,x,zero],[c,zero,x],[x,zero,c]]
	
	# each time search the h_1 in some range and put the value of bgr into the
	# corresponding index
	for i in range(6):
		idx = np.where((h_1>=i) & (h_1<i+1))
		bgr[:,:,0][idx] = (v-c)[idx] + b1g1r1[i][0][idx]
		bgr[:,:,1][idx] = (v-c)[idx] + b1g1r1[i][1][idx]
		bgr[:,:,2][idx] = (v-c)[idx] + b1g1r1[i][2][idx]		

	# all value out of 0 and 1 will become 0 and 1
	bgr = np.clip(bgr,0,1)
	bgr = (bgr*255).astype(np.uint8)
	return bgr



img = cv2.imread("imori.jpg").astype(np.float32)
# transfer bgr to hsv
hsv = bgr2hsv(img)

# reverse hue
hsv[:,:,0] = (hsv[:,:,0] + 180)%360

# transfer hsv to bgr
bgr = hsv2bgr(img, hsv)
cv2.imwrite("result5.jpg", bgr)
cv2.imshow("",bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()