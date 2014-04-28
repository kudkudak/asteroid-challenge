"""
File transforms files in data/* to data_aug/*.
In short - makes more images

Writes them aout to requested file
"""

import matplotlib.pylab as pl
from matplotlib import animation
import numpy as np
from skimage import exposure
import os
import json

from im_operators import *
import config

def load_img_det(i):
    raw_values = [float(x) for x in open("data/{0}_img.raw".format(i)).
        read().split(" ") if len(x) > 0]
    det = None
    with open("data/{0}.det".format(i)) as f:
        det = f.read().split(" ")
    return np.array(raw_values).reshape(4, 64, 64).astype("float64"), det



def preprocessing_gauss(img, det):
    return im_crop(ndimage.gaussian_filter(img / config.MaximumPixelIntensity, sigma=1.05),
                   6.0)

def preprocessing_gauss_smaller_gauss(img, det):
    return im_crop(ndimage.gaussian_filter(img / config.MaximumPixelIntensity , sigma=0.95),
                   6.0)


def preprocessing_no_gauss(img, det):
    return im_crop(img / config.MaximumPixelIntensity ,
                   6.0)
def preprocessing_no_gauss_2x(img, det):

    return im_crop(img / config.MaximumPixelIntensity,
                   2.0)


def preprocessing_no_gauss_1x(img, det):
    return im_crop(img / config.MaximumPixelIntensity,
                   1.0)

def preprocessing_no_gauss_3x(img, det):
    return im_crop(img / config.MaximumPixelIntensity,
                   3.0)

def preprocessing_no_gauss_4x(img, det):
    return im_crop(img / config.MaximumPixelIntensity,
                   4.0)

def preprocessing_gauss_eq(img, det):
    return im_crop(exposure.equalize_hist(ndimage.gaussian_filter(img, sigma=1.1)),
                   4.0)

def preprocessing_gauss_eq_leave_2x(img, det):
    """
    2x bigger (32x32) - to allow for rotation and cropping later (random_transformation_generator)
    """
    return im_crop(exposure.equalize_hist(ndimage.gaussian_filter(img, sigma=1.1)),
                   3.0)

def affine_add(img):
    return img + 0.3

def generator_crop_flip_8fold(img, det,  preprocessor=preprocessing_gauss_eq):
    # Preprocess image
    img = preprocessor(img, det)
    # Split
    img_rot = im_rotate(img, 45.0)
    img_rot_flip_1_1 = im_flip(img_rot, True, True)
    img_rot_flip_0_1 = im_flip(img_rot, False, True)
    img_rot_flip_1_0 = im_flip(img_rot, True, False)
    img_flip_1_1 = im_flip(img, True, True)
    img_flip_1_0 = im_flip(img, True, False)
    img_flip_0_1 = im_flip(img, False, True)
    # if int(det[-1])==0:
    #     print img
    #     import matplotlib.pylab as plt
    #     plt.imshow(img)
    #     plt.show()
    #
    #     print img_rot

    return [img, img_flip_1_1,img_flip_1_0 , img_flip_0_1, img_rot, img_rot_flip_1_1, img_rot_flip_0_1, img_rot_flip_1_0]


"""
Main workhorse for generating augumented dataset

Poorly written ;)

@note: CHANGE FOLD OUT IF YOU CHANGE GENERATOR!

@note: distnction into positive and negative chunks is not needed - it is plain stupid.
Chunks are only for HDD storage efficiency in fact - we will generate chunks anyway differently
Because data fits in RAM

IT should be around 1-2GB RAM Only

Make sure that chunk_size is divisible by fold_out and rather do not turn on divide
"""
def generate_aug(generator, preprocessor, chunk_size, folder=config.DataAugDir, prefix="data_chunk_",
                 fold_out = 8, divide=False, crop_factor = 2.0, difference=False,
                 ):
    assert chunk_size % fold_out == 0

    #Split to chunks positive and negatives

    # Check size
    im_list, det = load_img_det(1)
    im_test = preprocessor(im_list[0], det)

    print "Producing images sized ",im_test.shape

    # Prepare data
    chunk_positive = np.zeros(shape=(chunk_size, 4, im_test.shape[0],im_test.shape[1]))
    chunk_false = np.zeros(shape=(chunk_size, 4, im_test.shape[0],im_test.shape[1]))
    chunk_false_id = 0
    chunk_positive_id = 0
    chunk_count = 0
    chunk_false_dets = []
    chunk_positive_dets = []

    #
    ## Pregenerate ids
    #ids = range(chunk_size)

    for i in xrange(1, config.rawdataset_size+1):
        im_list, det = load_img_det(i)



        im_0_gen = generator(im_list[0], det, preprocessor)
        im_1_gen = generator(im_list[1], det, preprocessor)
        im_2_gen = generator(im_list[2], det, preprocessor)
        im_3_gen = generator(im_list[3], det, preprocessor)

        for im0, im1, im2, im3 in zip(im_0_gen, im_1_gen, im_2_gen, im_3_gen):

            if int(det[-1])==0:
                print im0
                print im1
                print im2
                print im3

            if difference:
                im1 -= im0
                im2 -= im0
                im3 -= im0

            if det[-1] == "1" or not divide:
                # Prepare


                chunk_positive[chunk_positive_id,0,:,:] = im0
                chunk_positive[chunk_positive_id,1,:,:] = im1
                chunk_positive[chunk_positive_id,2,:,:] = im2
                chunk_positive[chunk_positive_id,3,:,:] = im3
                chunk_positive_dets.append(det)
                chunk_positive_id += 1

                ### Write out chunk
                if chunk_positive_id == chunk_size:
                    print "Generated chunk ", chunk_count

                    for d in chunk_positive_dets:
                        assert d[-1] == "1" or not divide
                    assert len(chunk_positive_dets) == chunk_size
                    #
                    ## Random shuffle chunk
                    #if random_shuffle_chunk:
                    #    ids = np.random.choice(ids, chunk_size, replace=False)


                    np.save(os.path.join(folder, "p_"+prefix+str(chunk_count)+".npy"), chunk_positive)
                    with open(os.path.join(folder, "p_"+prefix+str(chunk_count)+"_dets.json"), "w") as f:
                        f.write(json.dumps(chunk_positive_dets))
                    chunk_positive_id = 0
                    chunk_positive_dets = []
                    chunk_count += 1
            else:
                 # Prepare

                chunk_false[chunk_false_id,0,:,:] = im0
                chunk_false[chunk_false_id,1,:,:] = im1
                chunk_false[chunk_false_id,2,:,:] = im2
                chunk_false[chunk_false_id,3,:,:] = im3
                chunk_false_dets.append(det)
                chunk_false_id += 1

                ### Write out chunk
                if chunk_false_id == chunk_size:
                    print "Generated chunk ", chunk_count

                    for d in chunk_false_dets:
                        assert d[-1] == "0"
                    assert len(chunk_false_dets) == chunk_size
                    #
                    #
                    ## Random shuffle chunk
                    #if random_shuffle_chunk:
                    #    ids = np.random.choice(ids, chunk_size, replace=False)

                    np.save(os.path.join(folder, "n_"+prefix+str(chunk_count)+".npy"), chunk_false)
                    with open(os.path.join(folder, "n_"+prefix+str(chunk_count)+"_dets.json"), "w") as f:
                        f.write(json.dumps(chunk_false_dets))
                    chunk_false_id = 0
                    chunk_false_dets = []
                    chunk_count += 1

    if chunk_positive_id > 0:
        ### Write out last chunks
        np.save(os.path.join(folder, "p_"+prefix+str(chunk_count)+".npy"), chunk_positive[0:chunk_positive_id])
        with open(os.path.join(folder, "p_"+prefix+str(chunk_count)+"_dets.json"), "w") as f:
            f.write(json.dumps(chunk_positive_dets))
        chunk_count += 1

    if chunk_false_id > 0:
        np.save(os.path.join(folder, "n_"+prefix+str(chunk_count)+".npy"), chunk_false[0:chunk_false_id])
        with open(os.path.join(folder,"n_"+ prefix+str(chunk_count)+"_dets.json"), "w") as f:
            f.write(json.dumps(chunk_false_dets))


    print "Generated ",chunk_count, " chunks"

    ### Write out desc
    desc = {"chunk_size": chunk_size, "fold_out": fold_out, "image_side":im_test.shape[0], "crop_factor":crop_factor}
    with open(os.path.join(folder, "data_aug.desc"), "w") as f:
        f.write(json.dumps(desc))


if __name__ == "__main__":

    generate_aug(generator_crop_flip_8fold, preprocessing_no_gauss_2x, chunk_size=160, crop_factor=2.0, difference=False)
