import pylab
import skimage
import skimage.io
import skimage.transform
import scipy.ndimage as ndimage
import os
import scipy
import matplotlib.pyplot as plt
from skimage.filter import roberts, sobel
from math import exp
import numpy as np

########################
###Filtered gradient:###
########################

#1. Load an image
img = skimage.img_as_float(skimage.io.imread(os.getcwd() + '/inputImg.png'))
print(img.shape)

#pylab.imshow(g); pylab.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

def gaussian(x, mu, sigma):
  return exp( -(((x-mu)/(sigma))**2)/2.0 )

g = rgb2gray(img)

kernel_radius = 3 # for an 7x7 filter
sigma = 10 # for [-2*sigma, 2*sigma]
hkernel = [gaussian(x, kernel_radius, sigma) for x in range(2*kernel_radius+1)]
vkernel = [x for x in hkernel]
kernel2d = [[xh*xv for xh in hkernel] for xv in vkernel]
kernelsum = sum([sum(row) for row in kernel2d])
kernel2d = [[x/kernelsum for x in row] for row in kernel2d]
k = np.array(kernel2d)

#2. Find the x and y components of the gradient Fx and Fy of the image smoothed with a Gaussian.
blurImg = scipy.signal.convolve2d(g, k)
#plt.imshow(blurImg, cmap = plt.get_cmap('gray'));plt.show()


Kgx = np.array([[ -1, 0, 1], [-2,0,2], [-1,0,1]])

Kgy = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])

Fx = scipy.signal.convolve2d(blurImg, Kgx)

Fy = scipy.signal.convolve2d(blurImg, Kgy)

#plt.imshow(Fx, cmap = plt.get_cmap('gray')); plt.show()

#plt.imshow(dx, cmap = plt.get_cmap('gray'))
 
F = np.absolute(Fx) + np.absolute(Fy)




D = np.arctan2(Fx,Fy)
D = np.degrees(D)

plt.imshow(D, cmap = plt.get_cmap('gray')); plt.show()

#########################
#######QUESTION 2#########
#########################


degList = [0,45,90,135]


newD = [[0 for x in range(len(D[0]))] for y in range(len(D))]
newD = np.array(newD)
newD.shape

for x in range(D.shape[0]):
	for y in range(D.shape[1]):
		#print(D[x][y])
		if D[x][y] < 0:
			D[x][y] + 180
		m = min(degList, key=lambda x:abs(x-D[x][y]))
		newD[x][y] = m
		print m

#plt.imshow(newD, cmap = plt.get_cmap('gray'))









#3. Compute the edge strength F (the magnitude of the gradient) and edge orientation D = arctan(Fy/Fx) at each pixel.
