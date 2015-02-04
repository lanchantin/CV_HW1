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
img = skimage.img_as_float(skimage.io.imread(os.getcwd() + '/checker.png'))
print(img.shape)



######
##2.## Find the x and y components of the gradient Fx and Fy of the image smoothed with a Gaussian.
######

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

blurImg = scipy.signal.convolve2d(g, k)
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


######
##1.## Compute the covariance matrix C over a neighborhood around each point.
######

L = []
L_coords = []				

for x in range(D.shape[0]):
	for y in range(D.shape[1]):
		Ex = (Fx[x][y])**2
		Exy = (Fx[x][y])*(Fy[x][y])
		Ey = (Fy[x][y])**2
		try: 
			Ex = Ex + (Fx[x-1][y])**2
			Exy = Exy + (Fx[x-1][y])*(Fy[x-1][y])
			Ey = Ey + (Fy[x-1][y])**2
		except: pass

		try:
			Ex = Ex + (Fx[x-1][y+1])**2
			Exy = Exy + (Fx[x-1][y+1])*(Fy[x-1][y+1])
			Ey = Ey + (Fy[x-1][y+1])**2
		except: pass

		try:
			Ex = Ex + (Fx[x][y+1])**2
			Exy = Exy + (Fx[x][y+1])*(Fy[x][y+1])
			Ey = Ey + (Fy[x][y+1])**2
		except: pass

		try: 
			Ex = Ex + (Fx[x+1][y+1])**2
			Exy = Exy + (Fx[x+1][y+1])*(Fy[x+1][y+1])
			Ey = Ey + (Fy[x+1][y+1])**2
		except: pass

		try: 
			Ex = Ex + (Fx[x+1][y])**2
			Exy = Exy + (Fx[x+1][y])*(Fy[x+1][y])
			Ey = Ey + (Fy[x+1][y])**2
		except: pass

		try: 
			Ex = Ex + (Fx[x+1][y-1])**2
			Exy = Exy + (Fx[x+1][y-1])*(Fy[x+1][y-1])
			Ey = Ey + (Fy[x+1][y-1])**2
		except: pass

		try: 
			Ex = Ex + (Fx[x][y-1])**2
			Exy = Exy + (Fx[x][y-1])*(Fy[x][y-1])
			Ey = Ey + (Fy[x][y-1])**2
		except: pass

		try: 
			Ex = Ex + (Fx[x-1][y-1])**2
			Exy = Exy + (Fx[x-1][y-1])*(Fy[x-1][y-1])
			Ey = Ey + (Fy[x-1][y-1])**2
		except: pass

		C = np.array([[Ex, Exy], [Exy, Ey]])

		eigVals = np.linalg.eigvals(C)
		smallEig = np.amin(eigVals)
		
		if smallEig > 0.01:
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
		try:
			iMask[x-1][y] = 1
			iMask[x-1][y+1] = 1
			iMask[x][y+1] = 1
			iMask[x+1][y+1] = 1
			iMask[x+1][y] = 1
			iMask[x+1][y-1] = 1
			iMask[x][y-1] = 1
			iMask[x-1][y-1] = 1
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

		# 	try: 
		# 		index2 = L_COORDS_SORTED.index([x-1,y+1])
		# 		if index2 > index:
		# 			L_COORDS_SORTED.pop(index2)
		# 	except: pass

		# 	try: 
		# 		index3 = L_COORDS_SORTED.index([x,y+1])
		# 		if index3 > index:
		# 			L_COORDS_SORTED.pop(index3)
		# 	except: pass

		# 	try: 
		# 		index4 = L_COORDS_SORTED.index([x+1,y+1])
		# 		if index5 > index:
		# 			L_COORDS_SORTED.pop(index5)
		# 	except: pass

		# 	try: 
		# 		index5 = L_COORDS_SORTED.index([x,y-1])
		# 		if index5 > index:
		# 			L_COORDS_SORTED.pop(index5)
		# 	except: pass

		# 	try: 
		# 		index6 = L_COORDS_SORTED.index([x+1,y-1])
		# 		if index6 > index:
		# 			L_COORDS_SORTED.pop(index6)
		# 	except: pass

		# 	try: 
		# 		index7 = L_COORDS_SORTED.index([x-1,y-1])
		# 		if index7 > index:
		# 			L_COORDS_SORTED.pop(index7)
		# 	except: pass
		# except: pass

iOutput = [[0 for x in range(len(D[0]))] for y in range(len(D))]
for i in range(0,len(L_COORD_OUTPUT)):
	x,y = L_COORD_OUTPUT[i]
	iOutput[x][y] = 1;




# [x-1][y]
# [x-1][y+1]
# [x][y+1]
# [x+1][y+1]
# [x+1][y]
# [x+1][y-1]
# [x][y-1]
# [x-1][y-1]

