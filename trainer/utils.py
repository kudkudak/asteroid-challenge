import time
import os
import cPickle

def timed(func):
    """ Decorator for easy time measurement """
    def timed(*args, **dict_args):
        tstart = time.time()
        result = func(*args, **dict_args)
        tend = time.time()
        print "{0} ({1}, {2}) took {3:2.4f} s to execute".format(func.__name__, len(args), len(dict_args), tend - tstart)
        return result

    return timed



cache_dict = {}
def cached_in_memory(func):
    global cache_dict


    def func_caching(*args, **dict_args):
        key = (func.__name__, args, frozenset(dict_args.items()))
        if key in cache_dict:
            return cache_dict[key]
        else:
            returned_value = func(*args, **dict_args)
            cache_dict[key] = returned_value
            return returned_value

    return func_caching
import pickle
CacheDirectory = "data_caches"
def cached_HDD(func):
    def func_caching(*args, **dict_args):
        key = ''.join([a for a in
                       str((func.__name__, args, frozenset(dict_args.items())))
                       if a in "abcdefghijklmnoprstuwyxcz1234567890_qwertyuiopasdfghjklzxcvbnm"])

        #TODO: add logger..
        print "HDDCACHEMODULE: Checking key ",key

        cache_file = os.path.join(CacheDirectory, "c"+str(key)+".cache.pkl")
        if os.path.exists(cache_file):
            print "HDDCACHEMODULE: Loading pickled file"
            return pickle.load(open(cache_file, "r"))
        else:
            returned_value = func(*args, **dict_args)
            pickle.dump(returned_value, open(cache_file,"w"))
            return returned_value

    return func_caching
