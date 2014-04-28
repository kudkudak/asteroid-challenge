import theano
import sys
import theano.tensor as T
from theano.tensor.nnet import conv
import numpy
import time





class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, weight_l1=0.0, weight_l2=0.0, activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                              layer
        """
        self.input = input

        self.weight_l1 = weight_l1
        self.weight_l2 = weight_l2

        W_values = numpy.asarray(rng.uniform(
                low=-numpy.sqrt(6. / (n_in + n_out)),
                high=numpy.sqrt(6. / (n_in + n_out)),
                size=(n_in, n_out)), dtype=theano.config.floatX)
        if activation == theano.tensor.nnet.sigmoid:
            W_values *= 4

        self.W = theano.shared(value=W_values, name='W')

        b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, name='b')


        self.output = activation(T.dot(input, self.W) + self.b)
        # parameters of the model
        self.params = [self.W, self.b]

    def regularization_cost(self):
        cost = 0
        if self.weight_l1 > 0:
            cost += self.weight_l1 * abs(self.W).sum()
        if self.weight_l2 > 0:
            cost += self.weight_l2 * (self.W*self.W).sum()
        return cost

class RegressionLayer(HiddenLayer):

    def __init__(self, rng, input, n_in, n_out, weight_l1=0.0, weight_l2=0.0, activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                              layer
        """
        super(RegressionLayer,self).__init__(rng, input, n_in, n_out, weight_l1, weight_l2, activation)

    def cost(self, y):
        err = y - self.output
        return T.mean((err * err).sum(axis=1))

    def errors(self, y):
        err = abs(y - self.output)
        return T.mean(err.sum(axis=1))
