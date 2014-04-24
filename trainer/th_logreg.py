import theano
import sys
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy
import time
from theano.ifelse import ifelse


class LogisticRegression(object):

    def __init__(self, input, n_in, n_out):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (e.g., one minibatch of input images)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoint lies

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the target lies
        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        self.W = theano.shared(value=numpy.zeros((n_in, n_out),
                                            dtype=theano.config.floatX), name='W' )
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=numpy.zeros((n_out,),
                                            dtype=theano.config.floatX), name='b' )

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)

        # compute prediction as class whose probability is maximal in
        # symbolic form
        self.y_pred=T.argmax(self.p_y_given_x, axis=1)

        self.params=[self.W, self.b]

    def negative_log_likelihood(self, y):
        """Return the mean of the negative log-likelihood of the prediction
        of this model under a given target distribution.

        .. math::

          \frac{1}{|\mathcal{D}|} \mathcal{L} (\theta=\{W,b\}, \mathcal{D}) =
          \frac{1}{|\mathcal{D}|} \sum_{i=0}^{|\mathcal{D}|} \log(P(Y=y^{(i)}|x^{(i)}, W,b)) \\
              \ell (\theta=\{W,b\}, \mathcal{D})


        :param y: corresponds to a vector that gives for each example the
                  correct label;

        Note: we use the mean instead of the sum so that
              the learning rate is less dependent on the batch size
        """
        # Trick with arange to extract correct labels probability
        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])




    #def precision(self, y):
    #    """Return a float representing the number of errors in the minibatch
    #    over the total number of examples of the minibatch ; zero
    #    one loss over the size of the minibatch
    #    """
    #
    #    def sum_precision(y_value_true, y_value_pred, sum):
    #        sum += y_value_true==0 and y_value_pred==y_value_pred
    #
    #    def sum_positive(y_value_true, sum):
    #        sum += y_value_true == 0
    #
    #
    #
    #    return T.scan

    def precision(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero
        one loss over the size of the minibatch
        """
        return T.mean(T.neq(self.y_pred[T.eq(y,0)], y[T.eq(y, 0)]))

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero
        one loss over the size of the minibatch
        """
        return T.mean(T.neq(self.y_pred, y))



### Parameters ###

learning_rate = 0.1
batch_size = 600
n_epochs = 100

from utils import *

if __name__=="__main__":

    #### Shared dataset objects ####
    datasets = load_data()

    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]


    ### Input and output ####
    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix()  # the data is presented as rasterized images (each being a 1-D row vector in x)
    y = T.ivector()  # the labels are presented as 1D vector of [long int] labels

    # construct the logistic regression class
    classifier = LogisticRegression(
                input=x.reshape((batch_size, 28 * 28)), n_in=28 * 28, n_out=10)

    # cost - quite standard
    cost = classifier.negative_log_likelihood(y)


    # error - again: compile EXPRESSION (not function)
    errors = classifier.errors(y)

    # :)
    # compute the gradient of cost with respect to theta = (W,b)
    # W and b shared
    g_W = T.grad(cost, classifier.W)
    g_b = T.grad(cost, classifier.b)


    # specify how to update the parameters of the model as a list of
    # (variable, update expression) pairs


    updates = [(classifier.W, classifier.W - learning_rate * g_W),
            (classifier.b, classifier.b - learning_rate * g_b)]


    #### Final functions ####

    # compiling a Theano function `train_model` that returns the cost, but in
    # the same time updates the parameter of the model based on the rules
    # defined in `updates`
    train_model = theano.function(inputs=[index],
            outputs=cost,
            updates=updates,
            givens={
                x: train_set_x[index * batch_size: (index + 1) * batch_size],
                y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    valid_model = theano.function(inputs=[index],
            outputs=errors,
            updates=updates,
            givens={
                x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    test_model = theano.function(inputs=[index],
            outputs=errors,
            updates=updates,
            givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})




    #############
    # Train Model #
    ##############

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0] / batch_size
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
                validation_losses = [valid_model(i)
                                     for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)

                print('epoch %i, minibatch %i/%i, validation error %f %%' % \
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

                    test_losses = [test_model(i)
                                   for i in xrange(n_test_batches)]
                    test_score = numpy.mean(test_losses)

                    print(('     epoch %i, minibatch %i/%i, test error of best'
                       ' model %f %%') %
                        (epoch, minibatch_index + 1, n_train_batches,
                         test_score * 100.))

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print(('Optimization complete with best validation score of %f %%,'
           'with test performance %f %%') %
                 (best_validation_loss * 100., test_score * 100.))
    print 'The code run for %d epochs, with %f epochs/sec' % (
        epoch, 1. * epoch / (end_time - start_time))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.1fs' % ((end_time - start_time)))