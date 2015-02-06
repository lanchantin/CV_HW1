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

octaves = 4
scales = 5
sigBase = 1.6


def GaussianKernelFunc(sigma):
    width = 1 + (int(6*sigma))
    kernel = np.zeros((width,width))
    k = 0
    for x in range(width):
        for y in range(width):
            kernel[x,y] = math.exp(-0.5 * ( math.pow((x-width/2)/sigma, 2.0) + math.pow((y-width/2)/sigma, 2.0)))/(2*math.pi*sigma*sigma)
            k += kernel[x,y]       
    for x in range(width):
        for y in range(width):
            kernel[x,y] /= k;      
    return kernel

###########################################################
#################### Load Images ##########################
###########################################################
img = skimage.img_as_float(skimage.io.imread(os.getcwd() + '/building.png'))

#Greyscale Image
I = np.dot(img[...,:3], [0.299, 0.587, 0.144])


###########################################################
##########Create Gaussian and DoG Pyramids ################
###########################################################

##### Gaussian Pyramid #####
print 'Creating Gaussian Pyramid...\n'
GaussPyramid = {}
for octave in range(octaves):
   # print '\n-----octave ',octave,'-----'
    for scale in range(scales+1):
        sigma = sigBase*(2.0**(octave+float(scale)/float(3)))
        #print 'scale ', scale,', sigma ', sigma
        gaussianKernel = GaussianKernelFunc(sigma)
        GaussPyramid[octave,scale] = scipy.signal.convolve2d(I,gaussianKernel,boundary='symm',mode='same')
    I = scipy.ndimage.interpolation.zoom(I,.5)


##### Diff of Gaussian (DoG) Pyramid #####
print 'Creating Difference of Gaussian Gaussian Pyramid...\n'
DoGPyramid = {}
for octave in range(octaves):
	for scale in range(scales):
		DoGPyramid[octave,scale] = np.subtract(GaussPyramid[octave,scale],GaussPyramid[octave,scale+1])


########################################################
################# COMPUTE ALL EXTREMA ##################
########################################################

##### Check to see if Extrema #####
def Extrema(x,y,octave,scale):
	notMin = False
	notMax = False
	for i in range(-1,2):
		if (notMax == False or notMin == False):
			for xi in range(-1,2):
				if (notMax == False or notMin == False):
					for yi in range(-1,2):
						if (notMax == False or notMin == False):
							try:
								if DoGPyramid[octave,scale+i][x+xi,y+yi] > DoGPyramid[octave,scale][x,y]:
									notMax = True
								elif DoGPyramid[octave,scale+i][x+xi,y+yi] < DoGPyramid[octave,scale][x,y]:
									notMin = True
							except: pass
	if (notMin == False or notMax == False):
		return True
	else:
		return False


r = 10
ExtremaCoords = []
ExtremaSigmas = []
filteredCoords = []
filteredSigmas = []
lowContrastThresh = 0.008
print 'Computing Extrema Values...\n'
for octave in range(0,octaves):
	for scale in range(1,(scales-1)):
		for x in range(0,len(DoGPyramid[octave,scale])):
			for y in range(0,len(DoGPyramid[octave,scale][0])):
				if Extrema(x,y,octave,scale):
					ExtremaCoords.append([x*(2**octave),y*(2**octave)])
					ExtremaSigmas.append(sigBase*(2.0**(octave+float(scale)/float(3))))

					#if math.fabs(DoGPyramid[octave,scale][x,y]) > lowContrastThresh:
					if (x-1 > 0) and (y-1 > 0) and (x+1 < DoGPyramid[octave,scale].shape[0]) and (y+1 < DoGPyramid[octave,scale].shape[1]):
						H_Dxx = DoGPyramid[octave,scale][x,y+1] + DoGPyramid[octave,scale][x,y-1] - 2.0 * DoGPyramid[octave,scale][x,y]
						H_Dyy = DoGPyramid[octave,scale][x+1,y] + DoGPyramid[octave,scale][x-1,y] - 2.0 * DoGPyramid[octave,scale][x,y]
						H_Dxy = (DoGPyramid[octave,scale][x+1,y+1]+DoGPyramid[octave,scale][x-1,y-1]-DoGPyramid[octave,scale][x-1,y+1]-DoGPyramid[octave,scale][x+1,y-1])/4.0
						TR = H_Dxx + H_Dyy
						Det = H_Dxx*H_Dyy - H_Dxy*H_Dxy

						if (Det >= 0) and (TR*TR/Det > (r+1.0)*(r+1.0)/r):
						    filteredCoords.append([x,y])
						    filteredSigmas.append(sigBase*(2.0**(octave+float(scale)/float(3))))


							     
########################################################
################### DISPLAY IMAGES #####################
########################################################

iMask = [[0 for x in range(len(img[0]))] for y in range(len(img))]
for i in range(0,len(ExtremaCoords)):
	x,y = ExtremaCoords[i]
	iMask[x][y] = 1

plt.imshow(iMask, cmap = plt.get_cmap('gray')); plt.show()


I = np.dot(img[...,:3], [0.299, 0.587, 0.144])
for i in range(0,len(ExtremaCoords)):
	x,y = ExtremaCoords[i]
	mag = (int(math.floor(ExtremaSigmas[i])/2))
	for xi in range(-mag, mag+1):
		for yi in range(-mag, mag+1):
			if (xi == mag) or (yi == mag) or (xi == -mag) or (yi == -mag):
				try:
					I[x+xi][y+yi] = 1
				except: pass
plt.imshow(I, cmap = plt.get_cmap('gray')); plt.show()



I = np.dot(img[...,:3], [0.299, 0.587, 0.144])
for i in range(0,len(filteredCoords)):
	x,y = filteredCoords[i]
	mag = (int(math.floor(filteredSigmas[i])/2))
	for xi in range(-mag, mag+1):
		for yi in range(-mag, mag+1):
			if (xi == mag) or (yi == mag) or (xi == -mag) or (yi == -mag):
				try:
					I[x+xi][y+yi] = 1
				except: pass
			elif (xi == mag-1) or (yi == mag-1) or (xi == -mag+1) or (yi == -mag+1):
				try:
					I[x+xi][y+yi] = 0
				except:
					pass
plt.imshow(I, cmap = plt.get_cmap('gray')); plt.show()



