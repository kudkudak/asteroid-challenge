""" File transforms files in data/* to data_aug/*.
In short - makes more images
"""
import matplotlib.pylab as pl
from matplotlib import animation
import numpy as np
from skimage import exposure
from im_operators import *
import os
from data_api import *
import json

def load_img_det(i):
    raw_values = [float(x) for x in open("data/{0}_img.raw".format(i)).
        read().split(" ") if len(x) > 0]
    det = None
    with open("data/{0}.det".format(i)) as f:
        det = f.read().split(" ")
    return np.array(raw_values).reshape(4, 64, 64), det


def preprocessing_gauss_eq(img, det):
    return im_crop(exposure.equalize_hist(ndimage.gaussian_filter(img, sigma=1.1)),
                   4.0)

def preprocessing_gauss_eq_leave_2x(img, det):
    """
    2x bigger (32x32) - to allow for rotation and cropping later (random_transformation_generator)
    """
    return im_crop(exposure.equalize_hist(ndimage.gaussian_filter(img, sigma=1.1)),
                   2.0)



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

    return [img, img_flip_1_1,img_flip_1_1 , img_flip_0_1, img_rot, img_rot_flip_1_1, img_rot_flip_0_1, img_rot_flip_1_0]


"""
Main workhorse for generating augumented dataset

Poorly written ;)

@note: CHANGE FOLD OUT IF YOU CHANGE GENERATOR!

@note: distnction into positive and negative chunks is not needed - it is plain stupid.
Chunks are only for HDD storage efficiency in fact - we will generate chunks anyway differently
Because data fits in RAM

IT should be around 1-2GB RAM Only
"""
def generate_aug(generator, preprocessor, chunk_size, folder=DataAugDir, prefix="data_chunk_", limit=100000000,
                 fold_out = 8
                 ):

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

    for i in xrange(1, rawdataset_size+1):
        if chunk_count > limit:
            break

        im_list, det = load_img_det(i)

        im_0_gen = generator(im_list[0], det, preprocessor)
        im_1_gen = generator(im_list[1], det, preprocessor)
        im_2_gen = generator(im_list[2], det, preprocessor)
        im_3_gen = generator(im_list[3], det, preprocessor)


        if det[-1] == "1":
            # Prepare
            for im0, im1, im2, im3 in zip(im_0_gen, im_1_gen, im_2_gen, im_3_gen):
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
                        assert d[-1] == "1"
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
            for im0, im1, im2, im3 in zip(im_0_gen, im_1_gen, im_2_gen, im_3_gen):
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

    ### Write out desc
    desc = {"chunk_size": chunk_size, "fold_out": fold_out, "image_side":im_test.shape[0]}
    with open("data_aug.desc", "w") as f:
        f.write(json.dumps(desc))


generate_aug(generator_crop_flip_8fold, preprocessing_gauss_eq, chunk_size=1000)
