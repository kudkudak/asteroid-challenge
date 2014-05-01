import os
import json
import numpy as np
import random
from utils import cached_HDD, cached_in_memory, timed
import skimage
import multiprocessing as mp
import time
import json
from itertools import cycle
from realtime_aug import *
from config import *



#####  Low level API #####

def load_img_det(i):
    raw_values = [float(x) for x in open("data/{0}_img.raw".format(i)).
        read().split(" ") if len(x) > 0]
    det = None
    with open("data/{0}.det".format(i)) as f:
        det = f.read().split(" ")
    return np.array(raw_values).reshape(ImageChannels, 64, 64), det



#
#for i in xrange(1, rawdataset_size + 1):
#    print "Reading ",i
#    img, det = load_img_det(i)
#    maximum_value = max(maximum_value, img.max())

aug_single_chunk_size = None # How big is one chunk ?
aug_fold_out = None # One image in raw dataset produces N images in augmented dataset
aug_image_side = None # Current image side in augumented dataset

try:

    augdataset_desc = json.loads(open(os.path.join(DataAugDir,"data_aug.desc")).read())
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


# Image size final
ImageSideFinal = int(aug_image_side // CROP_FACTOR) # Crop factor from real_aug


print "======================="
print "DATAAPI: Chunks count=", chunks_count, "so examples=", chunks_count*aug_single_chunk_size,\
    " Raw dataset size = ",rawdataset_size
print "DATAAPI: ImageSideFinal=", ImageSideFinal, " reduced from ", aug_image_side
print "==========================\n"

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
# Transformed to easy format
Y_in_memory = []
X_extra_in_memory = []

def get_example(id):
    """
    Fetches single example by loading chunk and loading example
    """
    c = get_chunk(id // aug_single_chunk_size)
    return c[0][id % aug_single_chunk_size].reshape((ImageChannels*aug_image_side**2)), c[1][id % aug_single_chunk_size]

from sklearn.preprocessing import normalize

def get_example_memory(id):
    """
    Fetches single example which are preloaded into RAM memory, that simple :) High level API

    Should be almost always called - will fit in 8GB
    """
    global X_in_memory, det_in_memory, Y_in_memory, dataset_in_memory, X_extra_in_memory


    if not dataset_in_memory:
        print "Loading chunks into memory.. stand still"
        dataset_in_memory = True

        # Create in-memory objects
        X_extra_in_memory = np.empty(shape=(rawdataset_size*aug_fold_out, ImportColumnCount), dtype="float32")
        X_in_memory = np.empty(shape=(rawdataset_size*aug_fold_out, ImageChannels * aug_image_side**2), dtype="float32")
        Y_in_memory = np.empty(shape=(rawdataset_size*aug_fold_out, ColumnsResultCoiunt), dtype="int32")



        # Load data
        in_memory_id = 0
        print "Chunks_count=", chunks_count
        for i in xrange(chunks_count):
            if i >= (chunks_count - 1): break
            chk, chk_det = get_chunk(i)[0], get_chunk(i)[1]


            for j in xrange(len(chk_det)):
                #print chk[j,:,:,:]
                X_in_memory[in_memory_id, :] = chk[j, :, :, :].reshape((ImageChannels* aug_image_side**2, ))

                X_extra_in_memory[in_memory_id, :] = [float(chk_det[j][id]) for id in ImportantColumns]
                Y_in_memory[in_memory_id, :] = [int(chk_det[j][id]) for id in ColumnsResult]
                in_memory_id += 1

        try:
            pass
            # Normalize X_extra_in_memory
            # TODO: add scikit-learn normalization
            #X_in_memory[0:in_memory_id,:] = normalize(X_in_memory[0:in_memory_id,:], axis=1, norm='l1')
        except:
            pass

        try:
            # Normalize X_extra_in_memory
            X_extra_in_memory = normalize(X_extra_in_memory, axis=1, norm='l1')
        except:
            pass


    return X_in_memory[id, :], Y_in_memory[id, :], X_extra_in_memory[id, :]


import numpy as np

@timed
@cached_HDD()
def get_training_test_matrices_bare(train_percentage=0.9, oversample_negative=False, limit_size = 10000000000,
                                    add_x_extra=True, generators=[], feature_gen=[]):
    """ Oversampling is useful if class are imbalanced """

    assert aug_fold_out is not None

    if oversample_negative is False:
        # Test and train ids without oversampling

        dataset_size = min(limit_size, rawdataset_size*aug_fold_out)
        dataset_chunk_number = min(chunks_count, dataset_size//aug_single_chunk_size + 1)


        # Chunks share the same fold out in almost all cases - pay attention to that
        train_chunk_ids = np.random.choice(dataset_chunk_number, int(dataset_chunk_number*train_percentage))
        train_ids = []
        for id in train_chunk_ids:
            if id*aug_single_chunk_size >= dataset_size-1: break
            train_ids += range(id*aug_single_chunk_size, min(dataset_size, (id+1)*aug_single_chunk_size))


        if oversample_negative:
            train_ids_negative = np.array([id for id in train_ids if Y_in_memory[id]==0])
            train_ids_positives = np.array([id for id in train_ids if Y_in_memory[id]==1])

            train_ids_positive_indexes = np.random.choice(len(train_ids_positives), int(0.5*dataset_size*train_percentage), replace=True)
            train_ids_negative_indexes = np.random.choice(len(train_ids_negative), int(0.5*dataset_size*train_percentage), replace=True)

            train_ids = list(train_ids_positives[train_ids_positive_indexes]) + list(train_ids_negative[train_ids_negative_indexes])

        random.shuffle(train_ids)


        set_of_ids = set(train_ids)

        test_ids = [id for id in xrange(dataset_size) if id not in set_of_ids]

        # Load into memory
        get_example_memory(0)

        # Normalized - ok
        if not add_x_extra:
            return X_in_memory[train_ids], Y_in_memory[train_ids, :], X_in_memory[test_ids, :], Y_in_memory[test_ids]
        else:
            #a = np.hstack((X_in_memory[train_ids, :], X_extra_in_memory[train_ids, :]))
            return np.hstack((X_in_memory[train_ids, :], X_extra_in_memory[train_ids, :])),\
                   Y_in_memory[train_ids],\
                    np.hstack((X_in_memory[test_ids, :], X_extra_in_memory[test_ids, :])),\
                   Y_in_memory[test_ids]





@timed
@cached_HDD()
def get_training_test_matrices_expanded(train_percentage=0.9, N = 100,
                                       add_x_extra=True, generator= default_generator,oversample_negative=False,  feature_gen=None, train_ids=None):
    """
    Gets expanded dataset
    """
    size_training = int(N*train_percentage)
    size_testing = N - size_training
    trn_iterator, tst_iterator = get_cycled_training_test_generators_bare(oversample_negative=oversample_negative, train_percentage=train_percentage,
                                                        add_x_extra=add_x_extra,
                                                        generator=generator,
                                                        feature_gen=feature_gen, train_ids=train_ids)


    # Fetch dynamically dimensions
    check_dim_example, check_dim_answer = next(trn_iterator)
    if train_percentage < 1.0:
        next(tst_iterator)

    X_train = np.empty(shape=(size_training, check_dim_example.shape[0]), dtype="float32")
    X_test = np.empty(shape=(size_testing, check_dim_example.shape[0]), dtype="float32")
    Y_train = np.empty(shape=(size_training, ColumnsResultCoiunt), dtype="float32")
    Y_test = np.empty(shape=(size_testing, ColumnsResultCoiunt), dtype="float32")

    print "Filling in training dataset"
    for id, (ex, label) in enumerate(trn_iterator):
        X_train[id, :] = ex
        Y_train[id] = label
        if id>=size_training-1: break

    print "Filling in testing dataset"
    for id, (ex, label) in enumerate(tst_iterator):
        X_test[id, :] = ex
        Y_test[id] = label
        if id>=size_testing-1: break

    return X_train, Y_train, X_test, Y_test


def generate_train_ids(limit_size=100000000000, train_percentage=0.9):
    # Test and train ids without oversampling
    dataset_size = min(limit_size, rawdataset_size*aug_fold_out)
    dataset_chunk_number = min(chunks_count, dataset_size//aug_single_chunk_size + 1)


    # Chunks share the same fold out in almost all cases - pay attention to that
    train_chunk_ids = np.random.choice(dataset_chunk_number, int(dataset_chunk_number*train_percentage))
    train_ids = []
    for id in train_chunk_ids:
        if id*aug_single_chunk_size >= dataset_size-1: break
        train_ids += range(id*aug_single_chunk_size, min(dataset_size, (id+1)*aug_single_chunk_size))
    return train_ids


def get_cycled_training_test_generators_bare(train_percentage=0.9, oversample_negative=False, limit_size = 10000000000,
                                       add_x_extra=True, generator = default_generator, feature_gen=None, train_ids=None):
    """ Oversampling is useful if class are imbalanced """

    assert aug_fold_out is not None

    # Load into memory
    get_example_memory(0)


    dataset_size = min(limit_size, rawdataset_size*aug_fold_out)
    dataset_chunk_number = min(chunks_count, dataset_size//aug_single_chunk_size + 1)


    if train_ids is None:
        train_ids = generate_train_ids(limit_size, train_percentage)

    if oversample_negative:
        train_ids_negative = np.array([id for id in train_ids if any(Y_in_memory[id]==0)])
        train_ids_positives = np.array([id for id in train_ids if any(Y_in_memory[id]==1)])

        train_ids_positive_indexes = np.random.choice(len(train_ids_positives), int(0.5*dataset_size*train_percentage), replace=True)
        train_ids_negative_indexes = np.random.choice(len(train_ids_negative), int(0.5*dataset_size*train_percentage), replace=True)

        train_ids = list(train_ids_positives[train_ids_positive_indexes]) + list(train_ids_negative[train_ids_negative_indexes])

    random.shuffle(train_ids)

    random.shuffle(train_ids)


    set_of_ids = set(train_ids)

    test_ids = [id for id in xrange(dataset_size) if id not in set_of_ids] if train_percentage < 1.0 else []

    random.shuffle(test_ids)


    train_generator, test_generator = None, None


    if add_x_extra:
        def train_generator():
            for i in cycle(train_ids):
                datum = get_example_memory(i)
                added_features = feature_gen(*datum) if feature_gen else []
                yield np.hstack(
                    (generator(datum[0].reshape(ImageChannels, aug_image_side, aug_image_side)).
                     reshape(-1), datum[2], added_features)), datum[1]

        def test_generator():
            for i in cycle(test_ids):
                datum = get_example_memory(i)
                added_features = feature_gen(*datum) if feature_gen else []
                yield np.hstack((default_generator(datum[0].reshape(ImageChannels, aug_image_side, aug_image_side)).
                     reshape(-1), datum[2], added_features)), datum[1]

    else:
        def train_generator():
            for i in cycle(train_ids):
                datum = get_example_memory(i)
                added_features = feature_gen(*datum) if feature_gen else []
                yield np.hstack(
                    (generator(datum[0].reshape(ImageChannels, aug_image_side, aug_image_side)).
                     reshape(-1), added_features)), datum[1]

        def test_generator():
            for i in cycle(test_ids):
                datum = get_example_memory(i)
                added_features = feature_gen(*datum) if feature_gen else []
                yield np.hstack((default_generator(datum[0].reshape(ImageChannels, aug_image_side, aug_image_side)).
                     reshape(-1), added_features)), datum[1]

    return train_generator(), test_generator()







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






#
#def realtime_augmented_data_gen(num_chunks=None, chunk_size=CHUNK_SIZE, augmentation_params=default_augmentation_params,
#                                ds_transforms=ds_transforms_default, target_sizes=None, processor_class=LoadAndProcess):
#    if target_sizes is None: # default to (53,53) for backwards compatibility
#        target_sizes = [(53, 53) for _ in xrange(len(ds_transforms))]
#
#    n = 0 # number of chunks yielded so far
#    while True:
#        if num_chunks is not None and n >= num_chunks:
#            # print "DEBUG: DATA GENERATION COMPLETED"
#            break
#
#        # start_time = time.time()
#        selected_indices = select_indices(num_train, chunk_size)
#        labels = y_train[selected_indices]
#
#        process_func = processor_class(ds_transforms, augmentation_params, target_sizes)
#
#        target_arrays = [np.empty((chunk_size, size_x, size_y, 3), dtype='float32') for size_x, size_y in target_sizes]
#        pool = mp.Pool(NUM_PROCESSES)
#        gen = pool.imap(process_func, selected_indices, chunksize=100) # lower chunksize seems to help to keep memory usage in check
#
#        for k, imgs in enumerate(gen):
#            # print ">>> converting data: %d" % k
#            for i, img in enumerate(imgs):
#                target_arrays[i][k] = img
#
#        pool.close()
#        pool.join()
#
#        # TODO: optionally do post-augmentation here
#
#        target_arrays.append(labels)
#
#        # duration = time.time() - start_time
#        # print "chunk generation took %.2f seconds" % duration
#
#        yield target_arrays, chunk_size
#
#        n += 1




