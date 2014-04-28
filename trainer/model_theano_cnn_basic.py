import sklearn
import sklearn.preprocessing
import cPickle
import theano
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy
from theano.tensor.signal import downsample
import time

from th_logreg import LogisticRegression
from th_hiddenlayer import HiddenLayer, RegressionLayer
from th_cnn import LeNetConvPoolLayer
from data_api import *
from realtime_aug import *
import data_api

learning_rate = 0.1
rng = numpy.random.RandomState(23455)
learning_rate = 0.01
batch_size = 20
learning_rate = 0.01
n_epochs = 2
N=150000
N=200000
add_extra = True
onlyLast=True
logReg = False
MODEL_NAME = "theano_cnn.pkl"



def shared_dataset(data_x, data_y, borrow=True):
    """ Function that loads the dataset into shared variables

    The reason we store our dataset in shared variables is to allow
    Theano to copy it into the GPU memory (when code is run on GPU).
    Since copying data into the GPU is slow, copying a minibatch everytime
    is needed (the default behaviour if the data is not in a shared
    variable) would lead to a large decrease in performance.
    """
    shared_x = theano.shared(numpy.asarray(data_x,
                                           dtype=theano.config.floatX),
                             borrow=borrow)
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
    return shared_x, T.cast(shared_y, 'int32')

if __name__ == "__main__":
    #### LOAD DATASET ###
    train_set_x, train_set_y, test_set_x, test_set_y = \
        get_training_test_matrices_expanded(N=N, oversample_negative=True, generator=generator_fast, add_x_extra=True)

    #### NORMALIZATION + scaling ######
    print "Normalizing"
    normalizer = sklearn.preprocessing.Normalizer(norm='l2', copy=False)
    normalizer.fit_transform(train_set_x)
    normalizer.transform(test_set_x)
    train_set_x*=100.0
    test_set_x*=100

    #### PREPARE OUTPUT LAYER layer #####
    if onlyLast:
        train_set_y = train_set_y[:,3].reshape(-1,1)
        test_set_y = test_set_y[:,3].reshape(-1,1)

    last_layer = 1 if len(train_set_y.shape)==1 else train_set_y.shape[1]

    if logReg:
        train_set_y = train_set_y.reshape(-1)
        test_set_y = test_set_y.reshape(-1)

    print "LAST LAYER SIZE", last_layer

    #### PREPARE DATASET #######
    print train_set_x[0]

    train_set_x_extra = train_set_x[:, train_set_x.shape[1]-ExtraColumns:]
    train_set_x = train_set_x[:, 0:train_set_x.shape[1]-ExtraColumns]
    test_set_x_extra = test_set_x[:, test_set_x.shape[1]-ExtraColumns:]
    test_set_x = test_set_x[:, 0:test_set_x.shape[1]-ExtraColumns]

    print "Training on ..."
    print train_set_x.shape
    print "Image side ", data_api.ImageSideFinal

    train_set_x, train_set_y = shared_dataset(train_set_x, train_set_y)
    test_set_x, test_set_y = shared_dataset(test_set_x, test_set_y)

    if add_extra:
        train_set_x_extra, w = shared_dataset(train_set_x_extra, None)
        test_set_x_extra, w = shared_dataset(test_set_x_extra, None)


    print train_set_x.shape
    print test_set_x.shape

    print data_api.ImageSideFinal
    print data_api.ImageChannels*data_api.ImageSideFinal*data_api.ImageSideFinal



    ### Input and output ####
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix()  # the data is presented as rasterized images (each being a 1-D row vector in x)
    x_extra = T.matrix()  # the data is presented as rasterized images (each being a 1-D row vector in x)
    y = T.imatrix()  # the labels are presented as 1D vector of [long int] labels


    rng = numpy.random.RandomState(23455)

    ##############################
    # BEGIN BUILDING ACTUAL MODE
    ##############################

    # Better 32->32 ---> 16x16

    # Reshape matrix of rasterized images of shape (batch_size,ImageSideFinalxImageSideFinal)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size,int(ImageChannels),int(ImageSideFinal), int(ImageSideFinal)))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (20,20,12,12)

    # (14,14)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, data_api.ImageChannels, data_api.ImageSideFinal, data_api.ImageSideFinal),
            filter_shape=(40, data_api.ImageChannels, 5, 5), poolsize=(2, 2))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12 - 5 + 1, 12 - 5 + 1)=(8, 8)
    # maxpooling reduces this further to (8/2,8/2) = (4, 4)
    # 4D output tensor is thus of shape (20,50,4,4)
    l1ims =  ( data_api.ImageSideFinal - 5 + 1)/2
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, 40, ( data_api.ImageSideFinal - 5 + 1)/2, ( data_api.ImageSideFinal - 5 + 1)/2),
            filter_shape=(50, 40, 5, 5), poolsize=(2, 2))







    layer2 = None
    layer3 = None

    if not add_extra:
        # Output (14,14) -> (5, 5)

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20, 32 * 4 * 4) = (20, 512)
        layer2_input = layer1.output.flatten(2)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(rng, input=layer2_input,
                            n_in=50 * ((l1ims-4)/2)**2, n_out=500, weight_l1=0.001,
                            activation=T.tanh)

        # classify the values of the fully-connected sigmoidal layer
        layer3 = RegressionLayer(rng, input=layer2.output, n_in=500, n_out=last_layer, weight_l1=0.001)

    else:
        # Output (14,14) -> (5, 5)

        # the HiddenLayer being fully-connected, it operates on 2D matrices of
        # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
        # This will generate a matrix of shape (20, 32 * 4 * 4) = (20, 512)
        layer2_input = T.horizontal_stack(layer1.output.flatten(2), x_extra)

        # construct a fully-connected sigmoidal layer
        layer2 = HiddenLayer(rng, input=layer2_input,weight_l1=0.001,
                            n_in=50 * ((l1ims-4)/2)**2 + ExtraColumns, n_out=500,
                            activation=T.tanh)

        # classify the values of the fully-connected sigmoidal layer
        layer3 = RegressionLayer(rng, input=layer2.output, n_in=500, n_out=last_layer, weight_l1=0.001)

    model = [layer0, layer1, layer2, layer3]

    # the cost we minimize during training is the negative log likelihood of
    # the model plus the regularization terms (L1 and L2); cost is expressed
    # here symbolically
    cost = layer3.cost(y)# + layer3.regularization_cost() + layer2.regularization_cost()


    # error - again: compile EXPRESSION (not function)
    errors = layer3.errors(y)


    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # compute the gradient of cost with respect to theta (stored in params)
    # the resulting gradients will be stored in a list gparams
    gparams = []
    for param in params:
        gparam  = T.grad(cost, param)
        gparams.append(gparam)


    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs
    updates = []
    # given two list the zip A = [a1, a2, a3, a4] and B = [b1, b2, b3, b4] of
    # same length, zip generates a list C of same size, where each element
    # is a pair formed from the two lists :
    #    C = [(a1, b1), (a2, b2), (a3, b3) , (a4, b4)]
    for param, gparam in zip(params, gparams):
        updates.append((param, param - learning_rate * gparam))




    train_model, test_model = None, None

    if not add_extra:
        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    y: train_set_y[index * batch_size: (index + 1) * batch_size]})



        test_model = theano.function(inputs=[index],
                outputs=errors,
                givens={
                    x: test_set_x[index * batch_size: (index + 1) * batch_size],
                    y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    else:
        # compiling a Theano function `train_model` that returns the cost, but in
        # the same time updates the parameter of the model based on the rules
        # defined in `updates`
        train_model = theano.function(inputs=[index],
                outputs=cost,
                updates=updates,
                givens={
                    x: train_set_x[index * batch_size: (index + 1) * batch_size],
                    x_extra: train_set_x_extra[index * batch_size: (index + 1) * batch_size],
                    y: train_set_y[index * batch_size: (index + 1) * batch_size]})



        test_model = theano.function(inputs=[index],
                outputs=errors,
                givens={
                    x: test_set_x[index * batch_size: (index + 1) * batch_size],
                    x_extra: test_set_x_extra[index * batch_size: (index + 1) * batch_size],
                    y: test_set_y[index * batch_size: (index + 1) * batch_size]})


    get_example_th = theano.function(inputs=[index],
                outputs=x,
                givens={
                    x: train_set_x[index:index+1]})


    get_example_th_tst = theano.function(inputs=[index],
                outputs=(x,y),
                givens={
                    x: test_set_x[index:index+1],
                    y: test_set_y[index:index+1]})


    from visualize import *
    import matplotlib.pylab as plt


    #############
    # Train Model #
    ##############

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_test_batches = test_set_x.get_value(borrow=True).shape[0] / batch_size

    print '... training the model'
    # early-stopping parameters
    patience = 5000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                                  # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                  # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
                                  # go through this many
                                  # minibatche before checking the network
                                  # on the validation set; in this case we
                                  # check every epoch
    if not os.path.exists(MODEL_NAME):
        best_params = None
        best_validation_loss = numpy.inf
        test_score = 0.
        start_time = time.clock()

        done_looping = False
        epoch = 0
        while (epoch < n_epochs) and (not done_looping):
            epoch = epoch + 1
            for minibatch_index in xrange(n_train_batches):

                minibatch_avg_cost = train_model(minibatch_index)
                # iteration number
                iter = (epoch - 1) * n_train_batches + minibatch_index

                if (iter + 1) % validation_frequency == 0:
                    # compute zero-one loss on validation set
                    validation_losses = []
                    for i in xrange(n_test_batches):
                        out = test_model(i)
                        validation_losses.append(out)
                        # total, correct = 0., 0.
                        # for y_pred_val, y_tru_val in zip (out[2], out[3]):
                        #     if y_tru_val == 0:
                        #         total += 1.
                        #         if y_pred_val == 0: correct += 1.

                        # if total > 0: validation_losses_prec.append(correct/total)

                    this_validation_loss = numpy.mean(validation_losses)



                    print('epoch %i, minibatch %i/%i, validation error (mean abs x-y in regression) %f %%' % \
                        (epoch, minibatch_index + 1, n_train_batches,
                        this_validation_loss * 100.))


                    # if we got the best validation score until now
                    if this_validation_loss < best_validation_loss:
                        #improve patience if loss improvement is good enough
                        if this_validation_loss < best_validation_loss *  \
                        improvement_threshold:
                            patience = max(patience, iter * patience_increase)

                        best_validation_loss = this_validation_loss
                        # test it on the test set

                        test_losses = [test_model(i) for i in xrange(n_test_batches)]
                        test_score = numpy.mean(test_losses)

                        print(('     epoch %i, minibatch %i/%i, test error of best'
                        ' model %f %%') %
                            (epoch, minibatch_index + 1, n_train_batches,
                            test_score * 100.))

                if patience <= iter:
                    print "Patience ",patience, "iter ",iter
                    done_looping = True
                    break

        end_time = time.clock()

        print 'The code run for %d epochs, with %f epochs/sec' % (
            epoch, 1. * epoch / (end_time - start_time))


        import cPickle
        cPickle.dump(model, open(MODEL_NAME,"w"))
    else:
        layer0, layer1, layer2, layer3 = cPickle.load(open(MODEL_NAME, "r"))


    from visualize import show_4_ex

    from utils import tile_raster_images
    import matplotlib.pylab as plt
    #plt.imshow(tile_raster_images(
    #         X=layer0.W.get_value(borrow=True).T,
    #         img_shape=(data_api.ImageSideFinal, data_api.ImageSideFinal), tile_shape=(10, 10),
    #         tile_spacing=(1, 1)))
    #plt.show()
    show_4_ex(layer0.W.get_value()[0])
    show_4_ex(layer0.W.get_value()[1])

    try:
        for i in xrange(10000):
            xy = get_example_th_tst(i)[0]
            x = xy[0]
            y = xy[1]
            error = test_model(i)
            plt.imshow(x.reshape(ImageChannels, ImageSideFinal, ImageSideFinal)[0])
            plt.title("Label "+str(y)+" Error " + str(error))
            plt.show()
    except Exception, e:
        print str(e)

    try:
        

        ### Visualise weights
        from utils import tile_raster_images
        import matplotlib.pylab as plt
        plt.imshow(layour0.W.get_values(borrow=True).T[0].reshape(ImageChannels, ImageSideFinal, ImageSideFinal)[0])
        plt.show()
        plt.imshow(tile_raster_images(X=layer0.W.get_value(borrow=True).T,
                     img_shape=(max(ImageChannels/2,1)*data_api.ImageSideFinal, max(ImageChannels/2, 1)*data_api.ImageSideFinal),
                     tile_shape=(20, 20),
                     tile_spacing=(1, 1)))

        plt.show()
    except Exception,e:
        pass

