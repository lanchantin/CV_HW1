import pylab
import skimage
import skimage.io
import skimage.transform
import scipy.ndimage as ndimage
import os
import scipy
from scipy import linalg
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

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.144])


def GaussianKernel(sigma):
    width = 1 + 2*(int(3.0*sigma))
    mean = width/2
    kernel = np.zeros((width,width))
    sum = 0
    for x in range(width):
        for y in range(width):
            kernel[x,y] = math.exp(-0.5 * ( math.pow((x-mean)/sigma, 2.0) + math.pow((y-mean)/sigma, 2.0)))/(2*math.pi*sigma*sigma)
            sum += kernel[x,y]
    # normalize        
    for x in range(width):
        for y in range(width):
            kernel[x,y] /= sum;      
    return kernel


img = skimage.img_as_float(skimage.io.imread(os.getcwd() + '/building.png'))

I = rgb2gray(img)


sigma = 0.5
width = 7
octaves = 4
scales = 5
sigBase = 1.6

scaleNum = 3
thresPeak = 0.04
thresEdge = 10


#I = scipy.ndimage.interpolation.zoom(I,2)
print 'Gaussian Pyramid'
GaussPyramid = {}
for octave in range(octaves):
    print '\n-----octave ',octave,'-----'
    for scale in range(scales+1):
        sigma = sigBase*(2.0**(octave+float(scale)/float(3)))
        print 'scale ', scale,', sigma ', sigma
        gaussianKernel = GaussianKernelFunc(sigma)
        GaussPyramid[octave,scale] = scipy.signal.convolve2d(I,gaussianKernel,boundary='symm',mode='same')
    I = scipy.ndimage.interpolation.zoom(I,.5)


print 'Create DoG Pyramid\n'
DoGPyramid = {}
for octave in range(octaves):
	for scale in range(scales):
		DoGPyramid[octave,scale] = np.subtract(GaussPyramid[octave,scale],GaussPyramid[octave,scale+1])



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


# SIFT - perform the sub-pixel estimation
def interpolate_extrema(DoGPyramid, o, s, y, x, scaleNum,thresPeak):
    imgHeight = DoGPyramid[o,s].shape[0]
    imgWidth = DoGPyramid[o,s].shape[1]
    it = 0
    while it < 5:
        hessian = hessianMatrix(DoGPyramid, o, s, y, x)
        hessianInvert = linalg.inv(hessian)
        derivative = derivativeD(DoGPyramid, o, s, y, x)
        offset = NP.dot(hessianInvert,derivative)
        offset = offset*-1.0
        offset_x = offset[0]
        offset_y = offset[1]
        offset_s = offset[2]

        if offset_x < 0.5 and offset_y < 0.5 and offset_s < 0.5:
            break
        else:
            x += round(offset_x)
            y += round(offset_y)
            s += round(offset_s)
            if s < 1 or s > scaleNum or y < SIFT_IMG_BORDER or y >= imgHeight - SIFT_IMG_BORDER or x < SIFT_IMG_BORDER or x >= imgWidth - SIFT_IMG_BORDER:
                return []
        it += 1

    offset = [offset_x, offset_y, offset_s]
    derivative = derivativeD(DoGPyramid, o, s, y, x)
    D_offset = DoGPyramid[o,s][y,x] + NP.dot(derivative,offset)*0.5
    # since D value is recomputed, so we should refilter it by the thresPeak
    if math.fabs(D_offset) < thresPeak/scaleNum:
        return []

    feature = [o,s,y,x,offset_s,offset_y,offset_x]
    return feature


#--------------------------------------------------------------
#--------------------------------------------------------------
pre_eliminate_thres = 0.5*thresPeak/scaleNum
localExtremaList = []
thresPeakList = []
thresPeakSigmas = []
interpExtremaList = []
edgeEliminatedList = []
print 'Find Extrema\n'
ExtremaCoords = []
ExtremaSigmas = []
for octave in range(0,octaves):
	for scale in range(1,4):
		for x in range(0,len(DoGPyramid[octave,scale])):
			for y in range(0,len(DoGPyramid[octave,scale][0])):
				if Extrema(x,y,octave,scale):
					#print 'octave: ',octave," scale: ",scale," --> [", x*(2**octave), "][",y*(2**octave),"]"
					ExtremaCoords.append([x*(2**octave),y*(2**octave)])
					ExtremaSigmas.append(sigBase*(2.0**(octave+float(scale)/float(3))))

					if math.fabs(DoGPyramid[octave,scale][x,y]) > pre_eliminate_thres:
						thresPeakList.append([x,y])
						thresPeakSigmas.append(sigBase*(2.0**(octave+float(scale)/float(3))))

						feature = interpolate_extrema(DoGPyramid,octave,scale,x,y,scaleNum,thresPeak)

#-------------------------------------------------------------
#-------------------------------------------------------------




I = rgb2gray(img)
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



I = rgb2gray(img)
for i in range(0,len(thresPeakList)):
	x,y = thresPeakList[i]
	mag = (int(math.floor(thresPeakSigmas[i])/2))
	for xi in range(-mag, mag+1):
		for yi in range(-mag, mag+1):
			if (xi == mag) or (yi == mag) or (xi == -mag) or (yi == -mag):
				try:
					I[x+xi][y+yi] = 1
				except: pass

plt.imshow(I, cmap = plt.get_cmap('gray')); plt.show()



D = [[0 for x in range(len(img[0]))] for y in range(len(img))]
for i in range(0,len(ExtremaCoords)):
	x,y = ExtremaCoords[i]
	D[x][y] = ExtremaSigmas[i]




