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
from config import *
import config


### Constants - low level API
augdataset_desc = json.loads(open(os.path.join(config.DataAugDir,"data_aug.desc")).read())
IMAGE_HEIGHT = IMAGE_WIDTH = int(augdataset_desc["image_side"])
CROP_FACTOR = int(augdataset_desc["crop_factor"])
aug_single_chunk_size = None # How big is one chunk ?
aug_fold_out = None # One image in raw dataset produces N images in augmented dataset
aug_image_side = None # Current image side in augumented dataset
try:
    augdataset_desc = json.loads(open(os.path.join(DataAugDir,"data_aug.desc")).read())
    aug_single_chunk_size = augdataset_desc["chunk_size"]
    aug_fold_out = augdataset_desc["fold_out"]
    aug_image_side = augdataset_desc["image_side"]
except Exception, e:
    print "Failed with ",e


print "CROP_FACTOR=",CROP_FACTOR

# Augument this much
default_augmentation_params = {
    'zoom_range': (0.95, 1.05),
    'rotation_range': (0, 360.0),
    "shear_range":(0.01,0.09),
    "translation_range":(0,0)
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

    res = np.empty(shape=(ImageChannels, IMAGE_WIDTH//CROP_FACTOR, IMAGE_HEIGHT//CROP_FACTOR))

    transf = _random_perturbation_transform(**default_augmentation_params)

    for i in xrange(ImageChannels):
        res[i,:,:] = transf(img[i,:,:])

    return res


def generator_fast(img):
    """
    Generator transforms 4xXxY -> 4xX/CROP_FACTORxY/CROP_FACTOR
    """
    tf = random_perturbation_transform(**default_augmentation_params)
    res = np.empty(shape=(ImageChannels, aug_image_side//CROP_FACTOR, aug_image_side//CROP_FACTOR))
    for i in xrange(ImageChannels):
        res[i,:,:] = im_crop(fast_warp(img[i,:,:],tf), CROP_FACTOR)
    return res

def fast_warp(img, tf, mode='reflect'):
    """
This wrapper function is about five times faster than skimage.transform.warp, for our use case.
"""
    m = tf._matrix
    return skimage.transform._warps_cy._warp_fast(img, m,  mode=mode)

def default_generator(img):
    res = np.empty(shape=(ImageChannels, IMAGE_WIDTH//CROP_FACTOR, IMAGE_HEIGHT//CROP_FACTOR))
    for i in xrange(ImageChannels):
        res[i,:,:] = im_crop(img[i,:,:], CROP_FACTOR)
    return res

# def fast_warp(img, tf, output_shape=(53,53), mode='reflect'):
#     """
# This wrapper function is about five times faster than skimage.transform.warp, for our use case.
# """
#     m = tf._matrix
#     img_wf = np.empty((output_shape[0], output_shape[1]), dtype='float32')
#     img_wf[...] = skimage.transform._warps_cy._warp_fast(img[...], m, output_shape=output_shape, mode=mode)
#     return img_wf

center_shift = np.array((aug_image_side, aug_image_side)) / 2. - 0.5
tform_center = skimage.transform.SimilarityTransform(translation=-center_shift)
tform_uncenter = skimage.transform.SimilarityTransform(translation=center_shift)

def build_augmentation_transform(zoom=1.0, rotation=0, shear=0):
    tform_augment = skimage.transform.AffineTransform(scale=(1/zoom, 1/zoom), rotation=np.deg2rad(rotation), shear=np.deg2rad(shear))
    tform = tform_center + tform_augment + tform_uncenter # shift to center, augment, shift back (for the rotation/shearing)
    return tform

def random_perturbation_transform(zoom_range, rotation_range, translation_range, shear_range):
    """
    Generates transforming function
    """
    rotation = int(np.random.uniform(*rotation_range)) # there is no post-augmentation, so full rotations here!
    shear = np.random.uniform(*shear_range)

    log_zoom_range = [np.log(z) for z in zoom_range]
    zoom = np.exp(np.random.uniform(*log_zoom_range)) # for a zoom factor this sampling approach makes more sense.

    return build_augmentation_transform(zoom=zoom, shear=shear, rotation=rotation)



if __name__ == "__main__":
    # Is fast as good as simple?


    from data_api import get_example
    from visualize import *
    aug = build_augmentation_transform(zoom=1.1, rotation=210, shear=0.1)#random_perturbation_transform(**default_augmentation_params)
    import numpy as np
    for i in xrange(300,1000):
        im, det = get_example(i)
        im1 = im.reshape(ImageChannels, aug_image_side, aug_image_side)[0]
        im1_transformed = skimage.transform.warp(im1, aug, output_shape=(im1.shape[0], im1.shape[1]), mode='reflect')
        im1_transformed2 = skimage.transform.warp(im1, aug, output_shape=(im1.shape[0], im1.shape[1]), mode='reflect')
        show_4_ex([im1, im1, im1_transformed, im1_transformed2], det )

    # Test differences
    im = [i for i in im.reshape(4,aug_image_side,aug_image_side)]
    show_4_ex([im[0], im[1]-im[0], im[2]-im[0], im[3]-im[0]], det )


    import time

    start = time.time()
    for i in xrange(1000):
        generator_simple(im.reshape(ImageChannels, aug_image_side, aug_image_side))
    print "Generator simple took ",time.time() - start

    start = time.time()
    for i in xrange(1000):
        generator_fast(im.reshape(ImageChannels, aug_image_side, aug_image_side))
    print "Generator fast took ",time.time() - start


