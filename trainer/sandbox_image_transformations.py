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

print maximum_value




im_list, det = load_img_det(100)

im_0_gen = generator_crop_flip_8fold(im_list[0], det, preprocessing_gauss_eq_center)

print im_0_gen[0]

show_4_ex(im_0_gen[0:4], det)