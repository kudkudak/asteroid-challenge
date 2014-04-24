import time
import os
import cPickle
import numpy

def timed(func):
    """ Decorator for easy time measurement """
    def timed(*args, **dict_args):
        tstart = time.time()
        result = func(*args, **dict_args)
        tend = time.time()
        print "{0} ({1}, {2}) took {3:2.4f} s to execute".format(func.__name__, len(args), len(dict_args), tend - tstart)
        return result

    return timed

def scale_to_unit_interval(ndar, eps=1e-8):
  """ Scales all values in the ndarray ndar to be between 0 and 1 """
  ndar = ndar.copy()
  ndar -= ndar.min()
  ndar *= 1.0 / (ndar.max() + eps)
  return ndar


def tile_raster_images(X, img_shape, tile_shape, tile_spacing=(0, 0),
                       scale_rows_to_unit_interval=True,
                       output_pixel_vals=True):
  """
  Transform an array with one flattened image per row, into an array in
  which images are reshaped and layed out like tiles on a floor.

  This function is useful for visualizing datasets whose rows are images,
  and also columns of matrices for transforming those rows
  (such as the first layer of a neural net).

  :type X: a 2-D ndarray or a tuple of 4 channels, elements of which can
  be 2-D ndarrays or None;
  :param X: a 2-D array in which every row is a flattened image.

  :type img_shape: tuple; (height, width)
  :param img_shape: the original shape of each image

  :type tile_shape: tuple; (rows, cols)
  :param tile_shape: the number of images to tile (rows, cols)

  :param output_pixel_vals: if output should be pixel values (i.e. int8
  values) or floats

  :param scale_rows_to_unit_interval: if the values need to be scaled before
  being plotted to [0,1] or not


  :returns: array suitable for viewing as an image.
  (See:`PIL.Image.fromarray`.)
  :rtype: a 2-d array with same dtype as X.

  """

  assert len(img_shape) == 2
  assert len(tile_shape) == 2
  assert len(tile_spacing) == 2

  # The expression below can be re-written in a more C style as
  # follows :
  #
  # out_shape = [0,0]
  # out_shape[0] = (img_shape[0] + tile_spacing[0]) * tile_shape[0] -
  #                tile_spacing[0]
  # out_shape[1] = (img_shape[1] + tile_spacing[1]) * tile_shape[1] -
  #                tile_spacing[1]
  out_shape = [(ishp + tsp) * tshp - tsp for ishp, tshp, tsp
                      in zip(img_shape, tile_shape, tile_spacing)]

  if isinstance(X, tuple):
      assert len(X) == 4
      # Create an output numpy ndarray to store the image
      if output_pixel_vals:
          out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype='uint8')
      else:
          out_array = numpy.zeros((out_shape[0], out_shape[1], 4), dtype=X.dtype)

      #colors default to 0, alpha defaults to 1 (opaque)
      if output_pixel_vals:
          channel_defaults = [0, 0, 0, 255]
      else:
          channel_defaults = [0., 0., 0., 1.]

      for i in xrange(4):
          if X[i] is None:
              # if channel is None, fill it with zeros of the correct
              # dtype
              out_array[:, :, i] = numpy.zeros(out_shape,
                      dtype='uint8' if output_pixel_vals else out_array.dtype
                      ) + channel_defaults[i]
          else:
              # use a recurrent call to compute the channel and store it
              # in the output
              out_array[:, :, i] = tile_raster_images(X[i], img_shape, tile_shape, tile_spacing, scale_rows_to_unit_interval, output_pixel_vals)
      return out_array

  else:
      # if we are dealing with only one channel
      H, W = img_shape
      Hs, Ws = tile_spacing

      # generate a matrix to store the output
      out_array = numpy.zeros(out_shape, dtype='uint8' if output_pixel_vals else X.dtype)


      for tile_row in xrange(tile_shape[0]):
          for tile_col in xrange(tile_shape[1]):
              if tile_row * tile_shape[1] + tile_col < X.shape[0]:
                  if scale_rows_to_unit_interval:
                      # if we should scale values to be between 0 and 1
                      # do this by calling the `scale_to_unit_interval`
                      # function
                      this_img = scale_to_unit_interval(X[tile_row * tile_shape[1] + tile_col].reshape(img_shape))
                  else:
                      this_img = X[tile_row * tile_shape[1] + tile_col].reshape(img_shape)
                  # add the slice to the corresponding position in the
                  # output array
                  out_array[
                      tile_row * (H+Hs): tile_row * (H + Hs) + H,
                      tile_col * (W+Ws): tile_col * (W + Ws) + W
                      ] \
                      = this_img * (255 if output_pixel_vals else 1)
      return out_array

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
import inspect
def cached_HDD(func):
    def func_caching(*args, **dict_args):
        print  #UUGLY HACK
        key = ''.join([a for a in
                       str((func.__name__, args, frozenset([x for x in dict_args.values() if str(x).find("0x") == -1])))
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
