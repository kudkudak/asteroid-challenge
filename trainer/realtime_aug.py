"""
Realtime augumentations

Largely inspired by https://github.com/benanne/kaggle-galaxies/blob/master/realtime_augmentation.py
Thanks!
"""

import numpy as np
import skimage
import multiprocessing as mp
import time
import json
import skimage.version

# Prepare important global variables reading data aug descriptor
IMAGE_HEIGHT = IMAGE_WIDTH = json.loads(open("data_aug.desc").read())["image_side"]
center_shift = np.array((IMAGE_HEIGHT, IMAGE_WIDTH)) / 2. - 0.5
tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)


# Augument this much
default_augmentation_params = {
    'zoom_range': (1.0, 1.1),
    'rotation_range': (0, 360),
    'shear_range': (0, 0),
    'translation_range': (-4, 4),
}



def fast_warp(img, tf, output_shape=(53,53), mode='reflect'):
    """
    This wrapper function is about five times faster than skimage.transform.warp, for our use case.
    """
    m = tf._matrix
    img_wf = np.empty((output_shape[0], output_shape[1], 3), dtype='float32')
    for k in xrange(3):
        img_wf[..., k] = skimage.transform._warps_cy._warp_fast(img[..., k], m, output_shape=output_shape, mode=mode)
    return img_wf



def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom, 1/zoom),
            rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)

    tform = tform_center + tform_augment + tform_uncenter # shift to center, augment, shift back (for the rotation/shearing)
    return tform



def build_ds_transform(ds_factor=1.0,
                       orig_size=(IMAGE_WIDTH, IMAGE_HEIGHT),  do_shift=True):
    """
    This version is a bit more 'correct', it mimics the skimage.transform.resize function.
    """
    rows, cols = orig_size
    trows, tcols = orig_size[0]*ds_factor, orig_size[1]*ds_factor
    col_scale = row_scale = ds_factor
    src_corners = np.array([[1, 1], [1, rows], [cols, rows]]) - 1
    dst_corners = np.zeros(src_corners.shape, dtype=np.double)
    # take into account that 0th pixel is at position (0.5, 0.5)
    dst_corners[:, 0] = col_scale * (src_corners[:, 0] + 0.5) - 0.5
    dst_corners[:, 1] = row_scale * (src_corners[:, 1] + 0.5) - 0.5

    tform_ds = skimage.transform.AffineTransform()
    tform_ds.estimate(src_corners, dst_corners)

    if do_shift:
        shift_x = cols / (2 * ds_factor) - tcols / 2.0
        shift_y = rows / (2 * ds_factor) - trows / 2.0
        tform_shift_ds = skimage.transform.SimilarityTransform(translation=(shift_x, shift_y))
        return tform_shift_ds + tform_ds
    else:
        return tform_ds


def random_perturbation_transform(zoom_range, rotation_range, shear_range, translation_range, do_flip=False):
    # random shift [-4, 4] - shift no longer needs to be integer!
    shift_x = np.random.uniform(*translation_range)
    shift_y = np.random.uniform(*translation_range)
    translation = (shift_x, shift_y)

    # random rotation [0, 360]
    rotation = np.random.uniform(*rotation_range) # there is no post-augmentation, so full rotations here!

    # random shear [0, 5]
    shear = np.random.uniform(*shear_range)

    # # flip
    if do_flip and (np.random.randint(2) > 0): # flip half of the time
        shear += 180
        rotation += 180
        # shear by 180 degrees is equivalent to rotation by 180 degrees + flip.
        # So after that we rotate it another 180 degrees to get just the flip.

    # random zoom [0.9, 1.1]
    # zoom = np.random.uniform(*zoom_range)
    log_zoom_range = [np.log(z) for z in zoom_range]
    zoom = np.exp(np.random.uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.
    # the range should be multiplicatively symmetric, so [1/1.1, 1.1] instead of [0.9, 1.1] makes more sense.

    return build_augmentation_transform(zoom, rotation, shear, translation)

def perturb_and_dscrop(img, ds_transform, augmentation_params, target_size=None):
    tform_augment = random_perturbation_transform(**augmentation_params)
    return fast_warp(img, ds_transform + tform_augment, output_shape=target_size, mode='reflect').astype('float64')



from data_api import get_example

im1, det = get_example(0)

tform_ds_cropped5 = build_ds_transform(2.0, target_size=(IMAGE_WIDTH//2, IMAGE_HEIGHT//2))
ds_transforms_default = tform_ds_cropped5

print det