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
import scipy.ndimage
import math

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

def GaussianKernel(sigma, width):
    kernel = np.zeros((width,width))
    sum = 0.0
    mean = width/2
    for x in range(width):
        for y in range(width):
            kernel[x,y] = math.exp(-0.5 * ( math.pow((x-mean)/sigma, 2.0) + math.pow((y-mean)/sigma, 2.0)))/(2*math.pi*sigma*sigma)
            sum += kernel[x,y]
    # normalize        
    for x in range(width):
        for y in range(width):
            kernel[x,y] /= sum;      
    return kernel




sigma = 0.5
width = 7
octaves = 4
scales = 5

sigBase = 1.6
kBase = 2.0**(1.0/scales)

currGaussian = {}
for i in range(0,scales+3):
	currGaussian[i] = g

GaussianPyramid = {}
for o in range(octaves):
    for s in range(scales+1):
        k = kBase**s
        sigma = sigBase*k
        gaussianKernel = GaussianKernel(sigma,width)
        currGaussian[s] = scipy.signal.convolve2d(currGaussian[s],gaussianKernel,boundary='symm',mode='same')
        GaussianPyramid[o,s] = currGaussian[s]
        currGaussian[s] = scipy.ndimage.interpolation.zoom(GaussianPyramid[o,s],.5)
        print 'scale'
    sigBase = sigBase*2
    print 'octave'

DoGPyramid = {}
for o in range(octaves):
	print 'octave'
	for s in range(scales):
		print 'scale'
		DoGPyramid[o,s] = scipy.signal.fftconvolve(np.subtract(GaussianPyramid[o,s],GaussianPyramid[o,s+1]),g)




#plt.imshow(DoGPyramid[0,0], cmap = plt.get_cmap('gray')); plt.show()

iMask = [[0 for x in range(len(img[0]))] for y in range(len(img))]
for o in range(0,octaves):
	#for s in range(1,4):
	s = 2
	for x in range(0,len(DoGPyramid[o,s])):
		for y in range(0,len(DoGPyramid[o,s][0])):
			notMin = False
			notMax = False
			for i in range(-1,2):
				if (notMax == False or notMin == False):
					for xi in range(-1,2):
						if (notMax == False or notMin == False):
							for yi in range(-1,2):
								if (notMax == False or notMin == False):
									try:
										if DoGPyramid[o,s+i][x+xi,y+yi] > DoGPyramid[o,s][x,y]:
											notMax = True
										elif DoGPyramid[o,s+i][x+xi,y+yi] < DoGPyramid[o,s][x,y]:
											notMin = True
									except: pass
			if notMin == False or notMax == False:
				iMask[x][y] = 1

print "done"

