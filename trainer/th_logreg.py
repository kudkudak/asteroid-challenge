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

    def ypred0(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero
        one loss over the size of the minibatch
        """
        return self.y_pred

    def y0(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero
        one loss over the size of the minibatch
        """

        return y

    def precision(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero
        one loss over the size of the minibatch
        """
        return T.mean(T.neq(self.y_pred[(y==0).__nonzero__()], y[(y==0).__nonzero__()]))

    def errors(self, y):
        """Return a float representing the number of errors in the minibatch
        over the total number of examples of the minibatch ; zero
        one loss over the size of the minibatch
        """
        return T.mean(T.neq(self.y_pred, y))



