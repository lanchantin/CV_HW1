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
import math
from scipy import linalg

########################
###Filtered gradient:###
########################

#convert rgb to grayscale
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

inputImg = skimage.img_as_float(skimage.io.imread(os.getcwd() + '/checker.png'))

I = rgb2gray(inputImg)

Gaussian = GaussianKernel(2)

try:
	blurImg = scipy.signal.convolve2d(I, Gaussian, mode = 'same',boundary = 'symm')
except:
	blurImg = scipy.signal.convolve2d(inputImg, Gaussian, mode = 'same',boundary = 'symm')
	pass
#plt.imshow(blurImg, cmap = plt.get_cmap('gray'));plt.show()

#Sobel filter values
Kgx = np.array([[ -1, 0, 1], [-2,0,2], [-1,0,1]])
Kgy = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])


Fx = scipy.signal.convolve2d(blurImg, Kgx, mode = 'same',boundary = 'symm')
Fy = scipy.signal.convolve2d(blurImg, Kgy, mode = 'same',boundary = 'symm')

#plt.imshow(Fx, cmap = plt.get_cmap('gray')); plt.show()
#plt.imshow(Fy, cmap = plt.get_cmap('gray'))
 
######
##3.## Compute the edge strength F (the magnitude of the gradient) and edge orientation D = arctan(Fy/Fx) at each pixel.
######
F = np.absolute(Fx) + np.absolute(Fy)


D = np.arctan2(Fx,Fy)
D = np.degrees(D)
#plt.imshow(D, cmap = plt.get_cmap('gray')); plt.show()



########################
####Finding Corners:####
########################

##1.## Compute the covariance matrix C over a neighborhood around each point.
######
eigValList = []
eigValCoordList = []	
winRange = 4
for x in range(I.shape[0]):
	for y in range(I.shape[1]):
		Ex = 0
		Ey = 0
		Exy = 0	
		for xi in range(-winRange,winRange+1):
			for yi in range(-winRange,winRange+1):
				try:
					Ex += (Fx[x+xi][y+yi])*(Fx[x+xi][y+yi])
					Exy += (Fx[x+xi][y+yi])*(Fy[x+xi][y+xi])
					Ey += (Fy[x+xi][y+yi])*(Fy[x+xi][y+yi])
				except: pass

		C = np.array([[Ex, Exy], [Exy, Ey]])

		eigVals = np.linalg.eigvals(C)
		smallEig = np.amin(eigVals)
		
		if smallEig > 0.7:
			eigValList.append(smallEig)
			eigValCoordList.append([x,y])



coordListSORTED = [x for (y,x) in sorted(zip(eigValList,eigValCoordList), reverse=True)]


iMask = [[0 for x in range(len(D[0]))] for y in range(len(D))]
L_COORD_OUTPUT = []
windowRange = 19
for i in range(0,len(coordListSORTED)):
	x,y = coordListSORTED[i]
		#For each point p, remove all points in the neighborhood of p that occur lower in L.
	if iMask[x][y] != 1:
		L_COORD_OUTPUT.append([x,y])
		iMask[x][y] = 1
		for xi in range(-windowRange,windowRange+1):
			for yi in range(-windowRange,windowRange+1):
				try:
					iMask[x+xi][y+yi] = 1
				except: pass


boxWidth = 2
boxWidth2 = boxWidth-1
for i in range(0,len(L_COORD_OUTPUT)):
	xi,yi = L_COORD_OUTPUT[i]
	for x in range(-boxWidth, boxWidth+1):
		for y in range(-boxWidth, boxWidth+1):
			if (x == boxWidth) or (y == boxWidth) or (x == -boxWidth) or (y == -boxWidth):
				try:
					I[x+xi][y+yi] = 1
				except:
					pass
			elif (x == boxWidth2) or (y == boxWidth2) or (x == -boxWidth2) or (y == -boxWidth2):
				try:
					I[x+xi][y+yi] = 0
				except:
					pass

plt.imshow(I, cmap = plt.get_cmap('gray')); plt.show()

