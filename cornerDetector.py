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


picture = 'Checker'
folder = os.getcwd()+'/'+picture
img = skimage.img_as_float(skimage.io.imread(folder +'/'+picture+ '.png'))

#greyscale img
I = np.dot(img[...,:3], [0.299, 0.587, 0.144])

def GaussianKernel(sigma):
    width = 1 + 2*(int(3.0*sigma))
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


Gaussian = GaussianKernel(2)

try:
	blurImg = scipy.signal.convolve2d(I, Gaussian, mode = 'same',boundary = 'symm')
except:
	I = img
	blurImg = scipy.signal.convolve2d(I, Gaussian, mode = 'same',boundary = 'symm')
#plt.imshow(blurImg, cmap = plt.get_cmap('gray'));plt.show()


#################FILTERED GRADIENT#################
print('Computing Filtered Gradient...')
## Sobel filter values
Kgx = np.array([[ -1, 0, 1], [-2,0,2], [-1,0,1]])
Kgy = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])

##2.## Compute Gradient
Fx = scipy.signal.convolve2d(blurImg, Kgx, mode = 'same',boundary = 'symm')
Fy = scipy.signal.convolve2d(blurImg, Kgy, mode = 'same',boundary = 'symm')
skimage.io.imsave(folder + '/horizontalGradient.png', Fx)
skimage.io.imsave(folder + '/verticalGradient.png', Fy)
#plt.imshow(Fx, cmap = plt.get_cmap('gray')); plt.show()
#plt.imshow(Fy, cmap = plt.get_cmap('gray'))

##3.## Edge strength F (the magnitude of the gradient) at each pixel
F = np.absolute(Fx) + np.absolute(Fy)
skimage.io.imsave(folder + '/gradientStrength.png', F)
#plt.imshow(F, cmap = plt.get_cmap('gray')); plt.show()

##3.## Edge orientation D = arctan(Fy/Fx) at each pixel
D = np.arctan2(Fx,Fy)
D = np.degrees(D)
#plt.imshow(D, cmap = plt.get_cmap('gray')); plt.show()



############FINDING CORNERS####################

##1.## Compute the covariance matrix C over a neighborhood around each point.
print('Computing Covariance Matrix and Eigenvalues...')
eigValList = []
eigValCoordList = []	
winRange = 2
for x in range(I.shape[0]):
	for y in range(I.shape[1]):
		Ex = 0; Ey = 0; Exy = 0	
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
		

		if smallEig > .5:
			eigValList.append(smallEig)
			eigValCoordList.append([x,y])


coordListSORTED = [x for (y,x) in sorted(zip(eigValList,eigValCoordList), reverse=True)]

print('Nonmaximum Suppression...')
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


#Display Smalles Eigenvalues
eigPlot = [[0 for x in range(len(D[0]))] for y in range(len(D))]
for i in range(0,len(eigValCoordList)):
	x,y = eigValCoordList[i]
	eigPlot[x][y] = eigValList[i]
scipy.misc.imsave(folder + '/eigenValues.png', eigPlot)



print('Display Image...')
boxWidth = 5
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
#plt.imshow(I, cmap = plt.get_cmap('gray')); plt.show()
scipy.misc.imsave(folder + '/corners.png', I)




