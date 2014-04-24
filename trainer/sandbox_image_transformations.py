"""
cjordansquire afaik all challenges that involve signal data (sound or vision) are now won by deep CNNs with ReLU + dropout.

cjordansquire for data with heterogeneously typed features, tree-based methods like GBRTs and RFs are still competitive.

cjordansquire And for sparse high dim data (e.g. text) penalized linear models are still a strong baseline.


Sandbox results: no filters might be better than filters ;p
"""


import matplotlib.pylab as pl
from matplotlib import animation
import numpy as np
from skimage import exposure
from im_operators import *
import os
from data_api import *
import json
from visualize import *
from generate_aug_data import *
from skimage import data
from skimage.filter.rank import entropy
from skimage.morphology import disk
from skimage.util import img_as_ubyte
print maximum_value
from skimage import data, filter, color
from skimage.transform import hough_circle
from skimage.feature import peak_local_max
from skimage.draw import circle_perimeter
from skimage.util import img_as_ubyte
from sklearn.metrics import matthews_corrcoef
from skimage.data import camera
from skimage.filter import roberts, sobel

def preprocessing_gauss_center_entr(img, det):
    return im_crop(entropy(ndimage.gaussian_filter((img / maximum_value), sigma=0.95)
                   , disk(6)), 6.0)

def preprocessing_gauss_center(img, det):
    return im_crop(ndimage.gaussian_filter((img / maximum_value), sigma=0.95)
                   , 6.0)


def preprocessing_center(img, det):
    return im_crop((img / maximum_value)
                   , 6.0)


im_list, det = load_img_det(1000)


im = im_list[1]

im_0_gen = generator_crop_flip_8fold(im_list[1], det, preprocessing_gauss_center_entr)
im_1_gen = generator_crop_flip_8fold(im_list[1], det, preprocessing_gauss_center)

sob = roberts(im)
#edges = preprocessing_center(sob, det)
#edges = filter.canny(im_list[0], sigma=3, low_threshold=0, high_threshold=1000)
# Detect two radii
hough_radii = np.arange(15, 30, 2)
hough_res = hough_circle(sob, hough_radii)

#centers = []
#accums = []
#radii = []
#
#for radius, h in zip(hough_radii, hough_res):
#    # For each radius, extract two circles
#    peaks = peak_local_max(h, num_peaks=2)
#    centers.extend(peaks)
#    accums.extend(h[peaks[:, 0], peaks[:, 1]])
#    radii.extend([radius, radius])



print im_1_gen[0]

show_4_ex([im_list[0], roberts(im_1_gen[1]), im_0_gen[0], im_1_gen[1]], det)