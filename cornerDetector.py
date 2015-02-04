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

########################
###Filtered gradient:###
########################

######
##1.## Load an image
######
img = skimage.img_as_float(skimage.io.imread(os.getcwd() + '/building.png'))


######
##2.## Find the x and y components of the gradient Fx and Fy of the image smoothed with a Gaussian.
######

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

Gaussian = GaussianKernel(11,7)

blurImg = scipy.signal.convolve2d(g, Gaussian)
#plt.imshow(blurImg, cmap = plt.get_cmap('gray'));plt.show()

#Sobel filter values
Kgx = np.array([[ -1, 0, 1], [-2,0,2], [-1,0,1]])
Kgy = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])


Fx = scipy.signal.convolve2d(blurImg, Kgx)
Fy = scipy.signal.convolve2d(blurImg, Kgy)

#plt.imshow(Fx, cmap = plt.get_cmap('gray')); plt.show()
#plt.imshow(dx, cmap = plt.get_cmap('gray'))
 
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
L = []
L_coords = []				

for x in range(D.shape[0]):
	for y in range(D.shape[1]):
		Ex = (Fx[x][y])**2
		Exy = (Fx[x][y])*(Fy[x][y])
		Ey = (Fy[x][y])**2

		for xi in range(-1,2):
			for yi in range(-1,2):
				try:
					Ex = Ex + (Fx[x+xi][y+yi])**2
					Exy = Exy + (Fx[x+xi][y+yi])*(Fy[x+xi][y+xi])
					Ey = Ey + (Fy[x+xi][y+yi])**2
				except: pass

		C = np.array([[Ex, Exy], [Exy, Ey]])

		eigVals = np.linalg.eigvals(C)
		smallEig = np.amin(eigVals)
		
		if smallEig > 0.5:
			L.append(smallEig)
			L_coords.append([x,y])


L_COORDS_SORTED = [x for (y,x) in sorted(zip(L,L_coords))]
L = sorted(L)

iMask = [[0 for x in range(len(D[0]))] for y in range(len(D))]

L_COORD_OUTPUT = []

for i in range(0,len(L_COORDS_SORTED)):
	x,y = L_COORDS_SORTED[i]
		#For each point p, remove all points in the neighborhood of p that occur lower in L.

	if iMask[x][y] != 1:
		L_COORD_OUTPUT.append([x,y])
		iMask[x][y] = 1
		for xi in range(-7,8):
			for yi in range(-7,8):
				try:
					iMask[x+xi][y+yi] = 1
				except: pass

		# try:
		# 	index = L_COORDS_SORTED.index([x,y])
		# 	try: 
		# 		index0 = L_COORDS_SORTED.index([x+1,y])
		# 		if index0 > index:
		# 			L_COORDS_SORTED.pop(index0)
		# 	except: pass

		# 	try: 
		# 		index1 = L_COORDS_SORTED.index([x-1,y])
		# 		if index1 > index:
		# 			L_COORDS_SORTED.pop(index1)
		# 	except: pass

iOutput = [[0 for x in range(len(D[0]))] for y in range(len(D))]
for i in range(0,len(L_COORD_OUTPUT)):
	x,y = L_COORD_OUTPUT[i]
	iOutput[x][y] = 1;


plt.imshow(iOutput, cmap = plt.get_cmap('gray')); plt.show()

