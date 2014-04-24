import matplotlib.pylab as pl
from matplotlib import animation
import numpy as np
from skimage import exposure
from im_operators import *
import os
from data_api import *
import json


im_list, det = load_img_det(i)

im_0_gen = generator(im_list[0], det, preprocessor)
im_1_gen = generator(im_list[1], det, preprocessor)
im_2_gen = generator(im_list[2], det, preprocessor)
im_3_gen = generator(im_list[3], det, preprocessor)