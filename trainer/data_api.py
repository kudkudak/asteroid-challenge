import os
import json
import numpy as np
import random

##### Constants #####
DataAugDir = "data_aug"
ImageChannels = 4
ImageSide = 64


#####  Low level API #####

rawdataset_files = [os.path.join("data", f) for f in next(os.walk("data"))[2] if f.endswith(".raw")]
rawdataset_size = len(rawdataset_files)

aug_single_chunk_size = None # How big is one chunk ?
aug_fold_out = None # One image in raw dataset produces N images in augmented dataset
aug_image_side = None # Current image side in augumented dataset

try:
    augdataset_desc = json.loads(open("data_aug.desc").read())
    aug_single_chunk_size = augdataset_desc["chunk_size"]
    aug_fold_out = augdataset_desc["fold_out"]
    aug_image_side = augdataset_desc["image_side"]
except:
    pass


chunk_files = [os.path.join(DataAugDir, f) for f in next(os.walk(DataAugDir))[2] if f.endswith(".npy")]

## Check if names are correct
assert all([len(f.split("_")) == 4  for f in next(os.walk(DataAugDir))[2] if f.endswith(".npy")] ) == True

# Chunks accessors
chunks_positive_ids = [int(f.split("_")[3].split(".")[0])
                 for f in next(os.walk(DataAugDir))[2] if f.startswith("p_") and f.endswith(".npy")]
chunks_negative_ids = [int(f.split("_")[3].split(".")[0])
                 for f in next(os.walk(DataAugDir))[2] if f.startswith("n_") and f.endswith(".npy")]

positive_chunks = [f for f in chunk_files]
chunks_count = len(chunk_files)



def get_chunk(id):
    f_positive = os.path.join(DataAugDir, "p_data_chunk_"+str(id)+".npy")
    f_negative = os.path.join(DataAugDir, "n_data_chunk_"+str(id)+".npy")
    f_positive_det = os.path.join(DataAugDir, "p_data_chunk_"+str(id)+"_dets.json")
    f_negative_det = os.path.join(DataAugDir, "n_data_chunk_"+str(id)+"_dets.json")
    if(os.path.exists(f_positive)):
        return np.load(f_positive), json.load(open(f_positive_det))
    else:
        return np.load(f_negative), json.load(open(f_negative_det))


def free_memory():
    global X_in_memory, det_in_memory, Y_in_memory, dataset_in_memory
    dataset_in_memory = False
    del X_in_memory
    del det_in_memory
    del Y_in_memory


##### High level API #########




dataset_in_memory = False
# Transformed as big matrix
X_in_memory = []
det_in_memory = []
# Transformed to easy format
Y_in_memory = []

def get_example(id):
    """
    Fetches single example by loading chunk and loading example
    """
    c = get_chunk(id // aug_single_chunk_size)
    return c[0][id % aug_single_chunk_size].reshape((ImageChannels*aug_image_side**2)), c[1][id % aug_single_chunk_size]


def get_example_memory(id):
    """
    Fetches single example which are preloaded into RAM memory, that simple :) High level API
    """
    global X_in_memory, det_in_memory, Y_in_memory, dataset_in_memory


    if not dataset_in_memory:
        print "Loading chunks into memory.. stand still"
        dataset_in_memory = True

        # Create in-memory objects
        X_in_memory = np.empty(shape=(rawdataset_size*aug_fold_out, ImageChannels * aug_image_side**2))
        Y_in_memory = np.empty(shape=(rawdataset_size*aug_fold_out, ))
        det_in_memory = []

        # Load data
        in_memory_id = 0
        for i in xrange(chunks_count):
            chk, chk_det = get_chunk(i)[0], get_chunk(i)[1]
            for j in xrange(len(chk_det)):
                X_in_memory[in_memory_id, :] = chk[j, :, :, :].reshape((4* aug_image_side**2, ))
                Y_in_memory[in_memory_id] = float(chk_det[j][-1])
                det_in_memory.append(chk_det[j])
                in_memory_id += 1



    return X_in_memory[id, :], Y_in_memory[id], det_in_memory[id]



def get_training_test_matrices_bare(train_percentage=0.9, oversample_negative=False, limit_size = 10000000000):
    """ Oversampling is useful if class are imbalanced """
    assert oversample_negative == False
    assert aug_fold_out is not None

    if oversample_negative is False:
        # Test and train ids without oversampling

        dataset_size = min(limit_size, rawdataset_size*aug_fold_out)

        train_ids = np.random.choice( dataset_size,
                                     int(dataset_size*train_percentage),
                    replace=False)

        set_of_ids = set(train_ids)

        test_ids = [id for id in xrange(rawdataset_size*aug_fold_out) if id not in set_of_ids]

        # Load into memory
        get_example_memory(0)

        return X_in_memory[train_ids], Y_in_memory[train_ids, :], X_in_memory[test_ids, :], Y_in_memory[test_ids]





def get_training_test_generators_bare(train_percentage=0.9, oversample_negative=False, limit_size = 10000000000):
    """ Oversampling is useful if class are imbalanced """
    assert oversample_negative == False
    assert aug_fold_out is not None

    if oversample_negative is False:
        # Test and train ids without oversampling



        dataset_size = min(limit_size, rawdataset_size*aug_fold_out)

        print "Size of dataset ",dataset_size

        train_ids = np.random.choice( dataset_size,
                                     int(dataset_size*train_percentage),
                    replace=False)

        set_of_ids = set(train_ids)

        test_ids = [id for id in xrange(rawdataset_size*aug_fold_out) if id not in set_of_ids]

        print train_ids

        def train_generator():
            for i in train_ids:
                yield get_example_memory(i)

        def test_generator():
            for i in test_ids:
                yield get_example_memory(i)

        return train_generator(), test_generator()







