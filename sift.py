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
from scipy import linalg


lowContrastThresh = 0.009
octaves = 4
scales = 5
initialSigma = 1.6

###########################################################
#################### Load Images ##########################
###########################################################
picture = 'Checker'
folder = os.getcwd()+'/'+picture
img = skimage.img_as_float(skimage.io.imread(folder +'/'+picture+ '.png'))

#Greyscale Image
I = np.dot(img[...,:3], [0.299, 0.587, 0.144])

try:
	I.shape[1]
except:
	I = img

###########################################################
##########Create Gaussian and DoG Pyramids ################
###########################################################
def createGaussian(sigma):
    w = 1 + (int(6*sigma))
    G = np.zeros((w,w))
    k = 0
    for x in range(w):
        for y in range(w):
            G[x,y] = math.exp(-0.5 * ( math.pow((x-w/2)/sigma, 2.0) + math.pow((y-w/2)/sigma, 2.0)))/(2*math.pi*sigma*sigma)
            k += G[x,y]       
    for x in range(w):
        for y in range(w):
            G[x,y] /= k;      
    return G


##### Gaussian Pyramid #####
print 'Creating Gaussian Pyramid...\n'
GaussPyramid = {}
for octave in range(octaves):
   # print '\n-----octave ',octave,'-----'
    for scale in range(scales+1):
        sigma = initialSigma*(2.0**(octave+float(scale)/float(3)))
        #print 'scale ', scale,', sigma ', sigma
        gaussian = createGaussian(sigma)
        GaussPyramid[octave,scale] = scipy.signal.convolve2d(I,gaussian,boundary='symm',mode='same')
    I = scipy.ndimage.interpolation.zoom(I,.5)


##### Diff of Gaussian (DoG) Pyramid #####
print 'Creating Difference of Gaussian Pyramid...\n'
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
							if DoGPyramid[octave,scale+i][x+xi,y+yi] > DoGPyramid[octave,scale][x,y]:
								notMax = True
							elif DoGPyramid[octave,scale+i][x+xi,y+yi] < DoGPyramid[octave,scale][x,y]:
								notMin = True
	if (notMin == False or notMax == False):
		return True
	else:
		return False


ExtremaCoords = []
ExtremaSigmas = []
filteredCoords = []
filteredSigmas = []
keyLocalCoords = []
keyLocalSigma = []
r = 10
print 'Computing Extrema Values...\n'
for octave in range(0,octaves):
	for scale in range(1,(scales-1)):
		for x in range(0,len(DoGPyramid[octave,scale])):
			for y in range(0,len(DoGPyramid[octave,scale][0])):
				if (x-1 > 0) and (y-1 > 0) and (x+1 < DoGPyramid[octave,scale].shape[0]) and (y+1 < DoGPyramid[octave,scale].shape[1]):
					if Extrema(x,y,octave,scale):
						ExtremaCoords.append([x*(2**octave),y*(2**octave)])
						ExtremaSigmas.append(initialSigma*(2.0**(octave+float(scale)/float(3))))


						### KEYPOINT LOCALIZATION ###
						Dxx = DoGPyramid[octave,scale][x,y+1] + DoGPyramid[octave,scale][x,y-1] - 2.0 * DoGPyramid[octave,scale][x,y]
						Dyy = DoGPyramid[octave,scale][x+1,y] + DoGPyramid[octave,scale][x-1,y] - 2.0 * DoGPyramid[octave,scale][x,y]
						Dss = DoGPyramid[octave,scale+1][x,y] + DoGPyramid[octave,scale-1][x,y] - 2.0 * DoGPyramid[octave,scale][x,y]
						Dxy = (DoGPyramid[octave,scale][x+1,y+1]+DoGPyramid[octave,scale][x-1,y-1]-DoGPyramid[octave,scale][x-1,y+1]-DoGPyramid[octave,scale][x+1,y-1])/4.0
						Dxs = (DoGPyramid[octave,scale+1][x,y+1]+DoGPyramid[octave,scale-1][x,y-1]-DoGPyramid[octave,scale-1][x,y+1]-DoGPyramid[octave,scale+1][x,y-1])/4.0
						Dys = (DoGPyramid[octave,scale+1][x+1,y]+DoGPyramid[octave,scale-1][x-1,y]-DoGPyramid[octave,scale-1][x+1,y]-DoGPyramid[octave,scale+1][x-1,y])/4.0
						Hess = [[Dxx,Dxy,Dxs],[Dxy,Dyy,Dys],[Dxs,Dys,Dss]]

						dx = (DoGPyramid[octave,scale][x,y+1] - DoGPyramid[octave,scale][x,y-1])/2.0
						dy = (DoGPyramid[octave,scale][x+1,y] - DoGPyramid[octave,scale][x-1,y])/2.0
						ds = (DoGPyramid[octave,scale+1][x,y] - DoGPyramid[octave,scale-1][x,y])/2.0

						try:
							oX, oY, oS = (-1)*(np.dot(linalg.inv(Hess),[dx,dy,ds]))
						except: 
							oX = 0; oY = 0; oS = 0


						keyLocalCoords.append([(x*(2**octave))+oX,(y*(2**octave))+oY])
						keyLocalSigma.append((initialSigma*(2.0**(octave+float(scale)/float(3))))+oS)


						#### LOW CONTRAST THRESHOLD ###
						if math.fabs(DoGPyramid[octave,scale][x,y]) > lowContrastThresh:
							### EDGE RESPONSE ELIMINATION ##
							H_Dxx = DoGPyramid[octave,scale][x,y+1] + DoGPyramid[octave,scale][x,y-1] - 2.0 * DoGPyramid[octave,scale][x,y]
							H_Dyy = DoGPyramid[octave,scale][x+1,y] + DoGPyramid[octave,scale][x-1,y] - 2.0 * DoGPyramid[octave,scale][x,y]
							H_Dxy = (DoGPyramid[octave,scale][x+1,y+1]+DoGPyramid[octave,scale][x-1,y-1]-DoGPyramid[octave,scale][x-1,y+1]-DoGPyramid[octave,scale][x+1,y-1])/4.0
							TR = H_Dxx + H_Dyy
							Det = H_Dxx*H_Dyy - H_Dxy*H_Dxy

							if (Det >= 0) and (TR*TR/Det > (r+1.0)*(r+1.0)/r):
							    filteredCoords.append([(x*(2**octave))+oX,(y*(2**octave))+oY])
							    filteredSigmas.append((initialSigma*(2.0**(octave+float(scale)/float(3))))+oS)




							     
########################################################
################### DISPLAY IMAGES #####################
########################################################


img = skimage.img_as_float(skimage.io.imread(folder +'/'+picture+ '.png'))
I = np.zeros((img.shape[0], img.shape[1]),dtype='float')
I = np.dot(img[...,:3], [0.299, 0.587, 0.144])
try: I.shape[1]
except: I = img
for i in range(0,len(ExtremaCoords)):
	x,y = ExtremaCoords[i]
	mag = (int(math.floor(ExtremaSigmas[i])/1.5))
	for xi in range(-mag, mag+1):
		for yi in range(-mag, mag+1):
			if (xi == mag) or (yi == mag) or (xi == -mag) or (yi == -mag):
				try:
					I[x+xi][y+yi] = 1
				except: pass
#plt.imshow(I, cmap = plt.get_cmap('gray')); plt.show()
scipy.misc.imsave(folder + '/extrema.png', I)



img = skimage.img_as_float(skimage.io.imread(folder +'/'+picture+ '.png'))
I = np.zeros((img.shape[0], img.shape[1]),dtype='float')
I = np.dot(img[...,:3], [0.299, 0.587, 0.144])
try: I.shape[1]
except: I = img
for i in range(0,len(filteredCoords)):
	x,y = filteredCoords[i]
	try:
		mag = (int(math.floor(filteredSigmas[i])/1.5))
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
	except: pass
#plt.imshow(I, cmap = plt.get_cmap('gray')); plt.show()
scipy.misc.imsave(folder + '/filteredLocalizedExtrema.png', I)



img = skimage.img_as_float(skimage.io.imread(folder +'/'+picture+ '.png'))
I = np.zeros((img.shape[0], img.shape[1]),dtype='float')
I = np.dot(img[...,:3], [0.299, 0.587, 0.144])
try: I.shape[1]
except: I = img
for i in range(0,len(keyLocalCoords)):
	x,y = keyLocalCoords[i]
	try:
		mag = (int(math.floor(keyLocalSigma[i])/1.5))
		for xi in range(-mag, mag+1):
			for yi in range(-mag, mag+1):
				if (xi == mag) or (yi == mag) or (xi == -mag) or (yi == -mag):
					try:
						I[x+xi][y+yi] = 1
					except: pass
	except: pass
#plt.imshow(I, cmap = plt.get_cmap('gray')); plt.show()
scipy.misc.imsave(folder + '/localizedExtrema.png', I)



