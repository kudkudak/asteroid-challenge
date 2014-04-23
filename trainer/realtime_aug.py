"""
Realtime augumentations

Largely inspired by https://github.com/benanne/kaggle-galaxies/blob/master/realtime_augmentation.py
Thanks!
"""

import numpy as np
import skimage
import multiprocessing as mp
import time

center_shift = np.array((IMAGE_HEIGHT, IMAGE_WIDTH)) / 2. - 0.5
tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)

def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom, 1/zoom), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    tform = tform_center + tform_augment + tform_uncenter # shift to center, augment, shift back (for the rotation/shearing)
    return tform

def build_ds_transform(ds_factor=1.0, orig_size=(424, 424), target_size=(53, 53), do_shift=True, subpixel_shift=False):
"""
This version is a bit more 'correct', it mimics the skimage.transform.resize function.
"""
    rows, cols = orig_size
    trows, tcols = target_size
    col_scale = row_scale = ds_factor
    src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
    dst_corners = np.zeros(src_corners.shape, dtype=np.double)
    # take into account that 0th pixel is at position (0.5, 0.5)
    dst_corners[:, 0] = col_scale * (src_corners[:, 0] + 0.5) - 0.5
    dst_corners[:, 1] = row_scale * (src_corners[:, 1] + 0.5) - 0.5

    tform_ds = skimage.transform.AffineTransform()
    tform_ds.estimate(src_corners, dst_corners)

    if do_shift:
        if subpixel_shift:
            # if this is true, we add an additional 'arbitrary' subpixel shift, which 'aligns'
            # the grid of the target image with the original image in such a way that the interpolation
            # is 'cleaner', i.e. groups of <ds_factor> pixels in the original image will map to
            # individual pixels in the resulting image.
            #
            # without this additional shift, and when the downsampling factor does not divide the image
            # size (like in the case of 424 and 3.0 for example), the grids will not be aligned, resulting
            # in 'smoother' looking images that lose more high frequency information.
            #
            # technically this additional shift is not 'correct' (we're not looking at the very center
            # of the image anymore), but it's always less than a pixel so it's not a big deal.
            #
            # in practice, we implement the subpixel shift by rounding down the orig_size to the
            # nearest multiple of the ds_factor. Of course, this only makes sense if the ds_factor
            # is an integer.

            cols = (cols // int(ds_factor)) * int(ds_factor)
            rows = (rows // int(ds_factor)) * int(ds_factor)
            # print "NEW ROWS, COLS: (%d,%d)" % (rows, cols)


        shift_x = cols / (2 * ds_factor) - tcols / 2.0
        shift_y = rows / (2 * ds_factor) - trows / 2.0
        tform_shift_ds = skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))
        return tform_shift_ds + tform_ds
    else:
        return tform_ds