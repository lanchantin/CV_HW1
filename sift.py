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

######
##1.## Load an image
######
img = skimage.img_as_float(skimage.io.imread(os.getcwd() + '/building.png'))
print(img.shape)

#pylab.imshow(g); pylab.show()

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])

g = rgb2gray(img)


def gaussian(x, mu, sigma):
  return exp( -(((x-mu)/(sigma))**2)/2.0 )
kernel_radius = 3 # for an 7x7 filter
sigma = 10 # for [-2*sigma, 2*sigma]
hkernel = [gaussian(x, kernel_radius, sigma) for x in range(2*kernel_radius+1)]
vkernel = [x for x in hkernel]
kernel2d = [[xh*xv for xh in hkernel] for xv in vkernel]
kernelsum = sum([sum(row) for row in kernel2d])
kernel2d = [[x/kernelsum for x in row] for row in kernel2d]
k = np.array(kernel2d)


G1 = scipy.signal.convolve2d(g, k)

sigma = 50 # for [-2*sigma, 2*sigma]
hkernel = [gaussian(x, kernel_radius, sigma) for x in range(2*kernel_radius+1)]
vkernel = [x for x in hkernel]
kernel2d = [[xh*xv for xh in hkernel] for xv in vkernel]
kernelsum = sum([sum(row) for row in kernel2d])
kernel2d = [[x/kernelsum for x in row] for row in kernel2d]
k = np.array(kernel2d)

G2 = scipy.signal.convolve2d(g, k)


D = scipy.signal.convolve2d((G2 - G1),g) 




