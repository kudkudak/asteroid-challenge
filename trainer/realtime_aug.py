"""
Realtime augumentations

Largely inspired by https://github.com/benanne/kaggle-galaxies/blob/master/realtime_augmentation.py
Thanks!


Do not CENTER them!
"""
import numpy as np
import skimage
import multiprocessing as mp
import time
import json
import skimage.transform
from im_operators import *

augdataset_desc = json.loads(open("data_aug.desc").read())
IMAGE_HEIGHT = IMAGE_WIDTH = augdataset_desc["image_side"]
CROP_FACTOR = 2.0

# Augument this much
default_augmentation_params = {
    'zoom_range': (0.9, 1.1),
    'rotation_range': (0, 360),
    "shear_range":(-0.2,0.2),
    'translation_range': (-0.2, 0.2)
}

def load_img_det(i):
    raw_values = [float(x) for x in open("data/{0}_img.raw".format(i)).
        read().split(" ") if len(x) > 0]
    det = None
    with open("data/{0}.det".format(i)) as f:
        det = f.read().split(" ")
    return np.array(raw_values).reshape(4, 64, 64), det

def fast_warp(img, tf, output_shape=(53,53), mode='reflect'):
    """
This wrapper function is about five times faster than skimage.transform.warp, for our use case.
"""
    return skimage.transform.warp(img, tf)
## TRANSFORMATIONS ##

center_shift = np.array((IMAGE_HEIGHT, IMAGE_WIDTH)) / 2. - 0.5
tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)

def build_augmentation_transform(zoom=1.0, rotation=0, shear=0, translation=(0, 0)):
    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom, 1/zoom), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear), translation=translation)
    tform = tform_center + tform_augment + tform_uncenter
    return tform


def _random_perturbation_transform(img, zoom_range, rotation_range, shear_range,  translation_range, do_flip=False):
    # random shift [-4, 4] - shift no longer needs to be integer!
    shift_x = np.random.uniform(*translation_range)
    shift_y = np.random.uniform(*translation_range)

    translation = (shift_x, shift_y)
    rotation = np.random.uniform(*rotation_range) # there is no post-augmentation, so full rotations here!
    shear = np.random.uniform(*shear_range)

    log_zoom_range = [np.log(z) for z in zoom_range]
    zoom = np.exp(np.random.uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.

    x = im_rescale(im_rotate(img, rotation), zoom)

    x = skimage.transform.warp(x, skimage.transform.AffineTransform(translation=translation, shear=shear))

    return im_crop(x, CROP_FACTOR)


"""
Exported generator to data_api
"""
def generator_simple(img):
    return _random_perturbation_transform(img, **default_augmentation_params)


if __name__ == "__main__":
    from data_api import get_example
    from visualize import *


    im1, det = get_example(200)


    
    im1 = im1.reshape(4, IMAGE_WIDTH, IMAGE_HEIGHT)[0]

    print im1
    show_4_ex([im1, im1, im1, im1],det )
    #
    #print perturb_and_dscrop(im1)

    im1_pet = _random_perturbation_transform(im1, **default_augmentation_params)
    im2_pet = _random_perturbation_transform(im1, **default_augmentation_params)
    im3_pet = _random_perturbation_transform(im1, **default_augmentation_params)

    print im1_pet.shape

    show_4_ex([ im2_pet, im1_pet, im3_pet, im_crop(im1, CROP_FACTOR)], det)
    #show_4_ex([ perturb_and_dscrop(im1),  perturb_and_dscrop(im1),  perturb_and_dscrop(im1),  perturb_and_dscrop(im1)], det)

    print det



