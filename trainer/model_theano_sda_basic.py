import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy
from theano.tensor.signal import downsample
import time
from theano.tensor.shared_randomstreams import RandomStreams

### My theano imports
from th_logreg import LogisticRegression
from th_hiddenlayer import HiddenLayer
from th_cnn import LeNetConvPoolLayer
from th_sda import dA


### My imports
from data_api import *
from realtime_aug import *
import data_api

learning_rate = 0.1
rng = numpy.random.RandomState(23455)


learning_rate = 0.01
batch_size = 100



learning_rate = 0.1
batch_size = 20



n_epochs = 100
L1_reg=0.000
L2_reg=0.0001
N=100000

# N=20000

add_extra = False

def shared_dataset(data_x, data_y, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype="float64"),
                             borrow=borrow)
    if data_y:
        shared_y = theano.shared(numpy.asarray(data_y,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
    # When storing data on the GPU it has to be stored as floats
    # therefore we will store the labels as ``floatX`` as well
    # (``shared_y`` does exactly that). But during our computations
    # we need them as ints (we use labels as index, and if they are
    # floats it doesn't make sense) therefore instead of returning
    # ``shared_y`` we will have to cast it to int. This little hack
    # lets ous get around this issue
    return shared_x, T.cast(shared_y, 'int32') if data_y else None

if __name__ == "__main__":
    #### Shared dataset objects ####
    train_set_x, train_set_y, test_set_x, test_set_y = \
        get_training_test_matrices_expanded(train_percentage=1.0, N=N, generator=default_generator, add_x_extra=True)


    train_set_x_extra = train_set_x[:, train_set_x.shape[1]-ExtraColumns:]
    train_set_x = train_set_x[:, 0:(train_set_x.shape[1]-ExtraColumns)/ImageChannels]
    test_set_x_extra = test_set_x[:, test_set_x.shape[1]-ExtraColumns:]
    test_set_x = test_set_x[:, 0:(test_set_x.shape[1]-ExtraColumns)/ImageChannels]
    ModelImageChannels = 1


    ### Input and output ####
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix()  # the data is presented as rasterized images (each being a 1-D row vector in x)
    x_extra = T.matrix()  # the data is presented as rasterized images (each being a 1-D row vector in x)
    y = T.ivector()  # the labels are presented as 1D vector of [long int] labels


    print "Training on ..."
    print train_set_x.shape
    print "Image side ", data_api.ImageSideFinal

    train_set_x, w = shared_dataset(train_set_x, None)
    test_set_x, w = shared_dataset(test_set_x, None)



    ######################
    # BUILDING THE MODEL #
    ######################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(numpy_rng=rng, theano_rng=theano_rng, input=x,
            n_visible=ModelImageChannels*data_api.ImageSideFinal*data_api.ImageSideFinal,
            n_hidden=50)

    cost, updates = da.get_cost_updates(corruption_level=0.3,
                                learning_rate=learning_rate)


    hidden_activ = theano.function([index], da.get_hidden_values(x),
         givens = {x: train_set_x[index:index+1, :]})

    reconstruct = theano.function([index], da.forward(x),
         givens = {x: train_set_x[index:index+1, :]})

    get_example_theano = theano.function([index], x,
         givens = {x: train_set_x[index:index+1, :]})

    train_da = theano.function([index], cost, updates=updates,
         givens = {x: train_set_x[index * batch_size: (index + 1) * batch_size]})

    start_time = time.clock()

    ############
    # TRAINING #
    ############



    # go through training epochs
    for epoch in xrange(n_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_epochs):
            c.append(train_da(batch_index))

        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock





    ###########
    # METRICS #
    ###########


    try:
        ### Visualise weights
        from utils import tile_raster_images
        import matplotlib.pylab as plt
        plt.imshow(tile_raster_images(X=da.W.get_value(borrow=True).T,
                     img_shape=(max(ModelImageChannels/2,1)*data_api.ImageSideFinal, max(ModelImageChannels/2, 1)*data_api.ImageSideFinal),
                     tile_shape=(20, 20),
                     tile_spacing=(1, 1)))

        plt.show()
    except Exception,e:
        pass


    # Visualise scattering
    N = 60000
    x_plt, y_plt, clr_plt = [0]*N, [0]*N, [0]*N
    for i in xrange(N):
        act = hidden_activ(i)
        x_plt[i] = act[0,0]
        y_plt[i] = act[0,1]
        clr_plt[i] = train_set_y[i]

    f, (ax1, ax2) = plt.subplots(2)
    ax2.scatter([x_plt[i] for i in xrange(N) if train_set_y[i] == 0], [y_plt[i] for i in xrange(N) if train_set_y[i] == 0], c= [0 for i in xrange(N) if train_set_y[i] == 0])
    ax1.scatter([x_plt[i] for i in xrange(N) if train_set_y[i] == 1], [y_plt[i] for i in xrange(N) if train_set_y[i] == 1], c= [1 for i in xrange(N) if train_set_y[i] == 1])
    plt.show()

    # Visualize reconstruction
    for i in xrange(10):
        from visualize import *
        print ModelImageChannels
        input_img = get_example_theano(i).reshape(ModelImageChannels, ImageSideFinal, ImageSideFinal)
        reconstructed_img = reconstruct(i).reshape(ModelImageChannels, ImageSideFinal, ImageSideFinal)
        print reconstructed_img[0][0:2]
        print reconstructed_img.shape
        print input_img.shape
        if ModelImageChannels > 1:
            show_4_ex([input_img[0], input_img[1], reconstructed_img[0], reconstructed_img[1]], title=train_set_y[i])
        else:
            show_4_ex([input_img[0], input_img[0], reconstructed_img[0], reconstructed_img[0]], title=train_set_y[i])
    import cPickle
    cPickle.dump(dA, open("dA.pkl","w"))


