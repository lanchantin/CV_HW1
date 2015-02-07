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

sigma = 3
T_h = 0.3
T_l = 0.15

picture = '/Building'
folder = os.getcwd()+picture
img = skimage.img_as_float(skimage.io.imread(folder +picture+ '.png'))
I = np.dot(img[...,:3], [0.299, 0.587, 0.144]) #Greyscale Image


# CREATE GAUSSIAN FOR CONVOLUTION
w = 1 + (int(6*sigma))
Gaussian = np.zeros((w,w))
k = 0
for x in range(w):
    for y in range(w):
        Gaussian[x,y] = math.exp(-0.5 * ( math.pow((x-w/2)/sigma, 2.0) + math.pow((y-w/2)/sigma, 2.0)))/(2*math.pi*sigma*sigma)
        k += G[x,y]       
for x in range(w):
    for y in range(w):
        Gaussian[x,y] /= k;     

#BLUR IMAGE
try:
	blurImg = scipy.signal.convolve2d(I, Gaussian, mode = 'same',boundary = 'symm')
except:
	I = img
	blurImg = scipy.signal.convolve2d(I, Gaussian, mode = 'same',boundary = 'symm')

#plt.imshow(blurImg, cmap = plt.get_cmap('gray'));plt.show()


#################FILTERED GRADIENT#################
#Sobel filter values
print('Computing Filtered Gradient...')
Kgx = np.array([[ -1, 0, 1], [-2,0,2], [-1,0,1]])
Kgy = np.array([[1,2,1], [0,0,0], [-1,-2,-1]])

Fx = scipy.signal.convolve2d(blurImg, Kgx,mode='same',boundary = 'symm')
Fy = scipy.signal.convolve2d(blurImg, Kgy,mode='same',boundary = 'symm')
skimage.io.imsave(folder + '/horizontalGradient.png', Fx)
skimage.io.imsave(folder + '/verticalGradient.png', Fy)
#plt.imshow(Fx, cmap = plt.get_cmap('gray')); plt.show()
#plt.imshow(dx, cmap = plt.get_cmap('gray'))
 
##3.## Compute the edge strength F (the magnitude of the gradient) and edge orientation D = arctan(Fy/Fx) at each pixel.
F = np.absolute(Fx) + np.absolute(Fy)
#plt.imshow(F, cmap = plt.get_cmap('gray')); plt.show()
skimage.io.imsave(folder + '/gradientStrength.png', F)


D = np.arctan2(Fy,Fx)
D = np.degrees(D)
#plt.imshow(F, cmap = plt.get_cmap('gray')); plt.show()


################################
####Nonmaximum suppression######
################################
D_star = np.array([[0 for x in range(len(D[0]))] for y in range(len(D))])

for x in range(D.shape[0]):
	for y in range(D.shape[1]):
		if D[x][y] < 0:
			D[x][y] = D[x][y] + 180
		D_star[x][y] = round(D[x][y]/45.0)
		D_star[x][y] = (D_star[x][y])*45.0

##2.## If the edge strength F(x,y) is smaller than at least one of its neighbors along D*, set I(x,y) = 0, else set I(x,y) = F(x,y)
I = [[0 for x in range(len(F[0]))] for y in range(len(F))]
for x in range(F.shape[0]):
	for y in range(F.shape[1]):
		if(D_star[x,y] == 0 or D_star[x,y] == 180):
			try:
				if(F[x][y] < F[x][y-1] or F[x][y] < F[x][y+1]):
					I[x][y] = 0
				else:
					I[x][y] = F[x][y]
			except:
				pass
		elif(D_star[x,y] == 45):
			try:
				if(F[x][y] < F[x+1][y-1] or F[x][y] < F[x-1][y+1]):
					I[x][y] = 0
				else:
					I[x][y] = F[x][y]
			except:
				pass
		elif(D_star[x,y] == 90):
			try:
				if(F[x][y] < F[x-1][y] or F[x][y] < F[x+1][y]):
					I[x][y] = 0
				else:
					I[x][y] = F[x][y]
			except:
				pass
		else:
			try:
				if(F[x][y] < F[x-1][y-1] or F[x][y] < F[x+1][y+1]):
					I[x][y] = 0
				else:
					I[x][y] = F[x][y]
			except:
				pass

#plt.imshow(I, cmap = plt.get_cmap('gray')); plt.show()
scipy.misc.imsave(folder + '/edgeStrength.png', I)


######################################
###### Hysteresis thresholding: ######
######################################


iMask = [[0 for x in range(len(D[0]))] for y in range(len(D))]				
stack = [] 

for x in range(F.shape[0]):
	for y in range(F.shape[1]):
		if (I[x][y] > T_h):
			stack.append([x,y])
			iMask[x][y] = 1

while len(stack) > 0:
	x,y = stack.pop()
	for xi in range(-1,2):
		for yi in range(-1,2):
			if (not (xi == 0 and yi == 0)):
				try:
					if iMask[x+xi][y+yi] != 1:
						if I[x+xi][y+yi] > T_l:
							stack.append([(x+xi),(y+yi)])
							iMask[x+xi][y+yi] = 1
				except: pass


#plt.imshow(iMask, cmap = plt.get_cmap('gray')); plt.show()
scipy.misc.imsave(folder + '/edges.png', iMask)