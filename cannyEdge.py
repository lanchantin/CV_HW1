import pylab
import skimage
import skimage.io
import skimage.transform
import os

I = skimage.img_as_float(skimage.io.imread(os.getcwd() + 'LakeGeorge.png'))