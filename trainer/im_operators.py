""" File used for generating data and loading
Code inspired by https://github.com/benanne/kaggle-galaxies
"""
import numpy as np
from scipy import ndimage
import glob
import itertools
import threading
import time
import skimage.transform
import skimage.io
import skimage.filter
import gzip
import os
import Queue
import multiprocessing as mp

"""
Detected around 90%
Undetected around 10%

Anomaly detector?

10435   896 - exact proportions. Not very good 91% - 9%
"""

def im_translate(img, shift_x, shift_y):
    ## this could probably be a lot easier... meh.
    # downsampling afterwards is recommended
    translate_img = np.zeros_like(img, dtype=img.dtype)

    if shift_x >= 0:
        slice_x_src = slice(None, img.shape[0] - shift_x, None)
        slice_x_tgt = slice(shift_x, None, None)
    else:
        slice_x_src = slice(- shift_x, None, None)
        slice_x_tgt = slice(None, img.shape[0] + shift_x, None)

    if shift_y >= 0:
        slice_y_src = slice(None, img.shape[1] - shift_y, None)
        slice_y_tgt = slice(shift_y, None, None)
    else:
        slice_y_src = slice(- shift_y, None, None)
        slice_y_tgt = slice(None, img.shape[1] + shift_y, None)

    translate_img[slice_x_tgt, slice_y_tgt] = img[slice_x_src, slice_y_src]

    return  translate_img


def im_flip(img, flip_h, flip_v):
    if flip_h:
        img = img[::-1]
    if flip_v:
        img = img[:, ::-1]
    return img

def im_rotate(img, angle):
    return skimage.transform.rotate(img, angle, mode='reflect')

def im_rescale(img, scale_factor):
    zoomed_img = np.zeros_like(img, dtype=img.dtype)
    zoomed = skimage.transform.rescale(img, scale_factor)

    if scale_factor >= 1.0:
        shift_x = (zoomed.shape[0] - img.shape[0]) // 2
        shift_y = (zoomed.shape[1] - img.shape[1]) // 2
        zoomed_img[:,:] = zoomed[shift_x:shift_x+img.shape[0], shift_y:shift_y+img.shape[1]]
    else:
        shift_x = (img.shape[0] - zoomed.shape[0]) // 2
        shift_y = (img.shape[1] - zoomed.shape[1]) // 2
        zoomed_img[shift_x:shift_x+zoomed.shape[0], shift_y:shift_y+zoomed.shape[1]] = zoomed

    return zoomed_img

def im_downsample(img, ds_factor):
    return img[::ds_factor, ::ds_factor]

def im_crop(img, ds_factor):
    size_x = img.shape[0]
    size_y = img.shape[1]

    cropped_size_x = img.shape[0] // ds_factor
    cropped_size_y = img.shape[1] // ds_factor

    shift_x = (size_x - cropped_size_x) // 2
    shift_y = (size_y - cropped_size_y) // 2

    return img[shift_x:shift_x+cropped_size_x, shift_y:shift_y+cropped_size_y]

def im_lcn(img, sigma_mean, sigma_std):
    """
based on matlab code by Guanglei Xiong, see http://www.mathworks.com/matlabcentral/fileexchange/8303-local-normalization
"""
    means = ndimage.gaussian_filter(img, sigma_mean)
    img_centered = img - means
    stds = np.sqrt(ndimage.gaussian_filter(img_centered**2, sigma_std))
    return img_centered / stds