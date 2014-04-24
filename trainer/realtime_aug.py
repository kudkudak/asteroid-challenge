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

### Constants - low level API
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




def _random_perturbation_transform(zoom_range, rotation_range, shear_range,  translation_range, do_flip=False):
    """
    Generates transforming function
    """
    # random shift [-4, 4] - shift no longer needs to be integer!
    shift_x = np.random.uniform(*translation_range)
    shift_y = np.random.uniform(*translation_range)

    translation = (shift_x, shift_y)
    rotation = np.random.uniform(*rotation_range) # there is no post-augmentation, so full rotations here!
    shear = np.random.uniform(*shear_range)

    log_zoom_range = [np.log(z) for z in zoom_range]
    zoom = np.exp(np.random.uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.

    def transf(x):
        x = im_rescale(im_rotate(x, rotation), zoom)
        x = skimage.transform.warp(x, skimage.transform.AffineTransform(translation=translation, shear=shear))
        return im_crop(x, CROP_FACTOR)

    return transf


"""
Exported generator to data_api
"""
def generator_simple(img):
    """
    Generator transforms 4xXxY -> 4xX/CROP_FACTORxY/CROP_FACTOR
    """

    res = np.empty(shape=(4, IMAGE_WIDTH//CROP_FACTOR, IMAGE_HEIGHT//CROP_FACTOR))

    transf = _random_perturbation_transform(**default_augmentation_params)

    for i in xrange(4):
        res[i,:,:] = transf(img[i,:,:])

    return res

def default_generator(img):
    res = np.empty(shape=(4, IMAGE_WIDTH//CROP_FACTOR, IMAGE_HEIGHT//CROP_FACTOR))
    for i in xrange(4):
        res[i,:,:] = im_crop(img[i,:,:], CROP_FACTOR)
    return res


if __name__ == "__main__":
    pass # not confogrming to api
    #from data_api import get_example
    #from visualize import *
    #
    #
    #im1, det = get_example(200)
    #
    #
    #
    #im1 = im1.reshape(4, IMAGE_WIDTH, IMAGE_HEIGHT)[0]
    #
    #print im1
    #show_4_ex([im1, im1, im1, im1],det )
    ##
    ##print perturb_and_dscrop(im1)
    #
    #im1_pet = _random_perturbation_transform(im1, **default_augmentation_params)
    #im2_pet = _random_perturbation_transform(im1, **default_augmentation_params)
    #im3_pet = _random_perturbation_transform(im1, **default_augmentation_params)
    #
    #print im1_pet.shape
    #
    #show_4_ex([ im2_pet, im1_pet, im3_pet, im_crop(im1, CROP_FACTOR)], det)
    ##show_4_ex([ perturb_and_dscrop(im1),  perturb_and_dscrop(im1),  perturb_and_dscrop(im1),  perturb_and_dscrop(im1)], det)
    #
    #print det
    #


