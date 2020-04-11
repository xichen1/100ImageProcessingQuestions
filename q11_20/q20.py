# draw the hist plot for the image using matplotlib
import cv2
import numpy as np
import matplotlib.pyplot as plt

img = cv2.imread("imori_dark.jpg").astype(np.uint8)


plt.hist(img.ravel(),255, rwidth=0.8,range=(0,255))
plt.savefig("res10_hist.jpg")

plt.show()
