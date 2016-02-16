# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 19:41:01 2016

Neural Network with one hidden unit layer

@author: Tadeze
"""
import numpy as np
import matplotlib.pylab as plt

"""
Activation factory generator class
"""


def sigmoid(x):
    if x > 35:
        x = 35
    if x < -35:
        x = -35
    return 1.0 / (1.0 + np.exp(-x))


def d_sigmoid(x):
    return sigmoid(x)(1 - sigmoid(x))


# relu
def relu(x):
    return 0 if x<0 else x


# drelu
def d_relu(x):
    if x > 0:
        return 1
    elif x == 0:
        return np.random.uniform(0, 1)
    else:
        return 0


class Neuron(object):
    def __init__(self, X, learning_rate, minbatch, momentum, n_hidden, output, epoch):
        self.learning_rate = learning_rate
        self.n_hidden = n_hidden
        self.minbatch = minbatch
        self.momentum = momentum
        self.N, self.dimension = np.shape(X)
        self.output = output
        self.epoch = epoch
        self.w1 = 0.01 * np.random.randn(self.n_hidden, self.dimension)  # + np.zeros([self.n_hidden,1]) #bias + weight
        self.w2 = 0.01 * np.random.randn(self.output, self.n_hidden)  # + np.zeros([1,output]) # bias + weight
        # bias initialization
        self.b1 = np.zeros((self.n_hidden, 1))
        self.b2 = np.zeros((self.output, 1))

    # Evaluate cross entrop loss function
    def evaluate_loss(self, score, pred):
        yl = len(pred)
        score=np.clip(score,1e-3,1-1e-3)  #clip for overflow of 1 or 0
        pred=np.clip(pred.T,1e-3,1-1e-3)
        return  -np.sum((score * np.log(pred) + (1 - score) * np.log(1 - pred)))

    def forward(self, x_train):

        l1 = np.dot(self.w1, x_train.T) + self.b1
        nonl_l1 =np.vectorize(relu)(l1)      #np.vectorize(self.act.act(), otypes=[np.float64])(l1)
        l2 = np.dot(self.w2, nonl_l1) + self.b2
        z = np.vectorize(sigmoid)(l2)        # np.vectorize(self.softmax.act(),otypes=[np.float64])(l2)  # output score for class 1
        return z, l1, l2
        # return mean cross entropy loss and prediction probabilities

    def backprop(self, xb, yb, l1, l2, zb):
        N = np.shape(yb)[0]
        # error from output layer
        l2_error = zb-yb
        # gradient of output layer
        nonl_l1 =np.vectorize(relu, otypes=[np.float64])(l1)        #np.vectorize(self.act.act(), otypes=[np.float64])(l1)
        dldw2 = nonl_l1.dot(l2_error).T /N  # gradient of w2

        # error from hidden unit layer
        l1_error = l2_error.dot(self.w2)
        # gradient  of

        dldw1_delta = l1_error.T * np.vectorize(d_relu, otypes=[np.float64])(l1)     #np.vectorize(self.act.dact(), otypes=[np.float64])(l1)
        dldw1 = dldw1_delta.dot(xb)/N

        # bias
        grad_b1 = np.mean(dldw1_delta, axis=1, keepdims=True)  #just 1xH average over
        grad_b2 = np.mean(l2_error,keepdims=True)     #just 1x1 bias for the hidden layer , average over batches.

        return dldw1, dldw2, grad_b1, grad_b2

    def predict(self, x_test, y_test):
        z, l1, l2 = self.forward(x_test)
        accuray = 1 - np.mean(np.abs(np.round(z) - y_test.T))
        loss = self.evaluate_loss(y_test, z)
        return accuray, loss/len(y_test)

    def train(self, x_train, y_train, test_data, test_label):

        # initialize weights and biase
        log = open('log_file_output.csv', 'w')
        train_size, dimension = np.shape(x_train)
        momentum_w1 = np.zeros_like(self.w1)
        momentum_w2 = np.zeros_like(self.w2)
        momentum_b1 = np.zeros_like(self.b1)
        momentum_b2 = np.zeros_like(self.b2)

        # number of minibatches

        for ep in xrange(self.epoch):
            # sample index
            n_sample = range(train_size)
            np.random.shuffle(n_sample)
            # random shuffle in every epch
            it = 0
            epoch_error = []

            while it < train_size - self.minbatch:
                # Forward pass for all batch size  and compute error
                # update w1 and w2 based on the error size
                # for i in range(it,it+self.minbatch-1):
                x = x_train[it:self.minbatch + it, ]
                y = y_train[it:self.minbatch + it]

                z, l1, l2 = self.forward(x)
                dw1, dw2, grad_b1, grad_b2 = self.backprop(x, y, l1, l2, z.T)

                momentum_w1 = self.momentum * momentum_w1 -self.learning_rate * dw1
                momentum_w2 = self.momentum * momentum_w2 - self.learning_rate * dw2

                momentum_b1 = self.momentum * momentum_b1 +self.learning_rate * grad_b1
                momentum_b2 = self.momentum * momentum_b2 + self.learning_rate * grad_b2

                self.w2 = self.w2 + momentum_w2
                self.w1 = self.w1 + momentum_w1
                self.b1 = self.b1 + momentum_b1
                self.b2 = self.b2 + momentum_b2

                it += self.minbatch


            training_err = self.predict(x_train, y_train)
            testing_err = self.predict(test_data, test_label)
            print (training_err, testing_err)
            print "\n"
            log.write( str(ep) + "," + str(training_err[0]) + "," +str(training_err[1])+',' +str(testing_err[0]+','+str(testing_err[1])))
            log.write("\n")

        log.close()

def main():
    # np.random.seed(1432)
    # input data
    import cPickle
    dict = cPickle.load(open("cifar/cifar_2class_py2.p", "rb"))
    train_label = dict["train_labels"]
    train_data = dict["train_data"]
    test_label = dict["test_labels"]
    test_data = dict["test_data"]

    scaling = lambda x: (x - np.min(x, axis=0)) / (np.max(x, axis=0) - np.min(x, axis=0))
    normalize = lambda x: (x - np.mean(x, axis=0)) / (np.std(x, axis=0))
    X_train = train_data #[1:5000,:]
    Y_train = train_label#[1:5000] #[1:5000]
    X_test = test_data
    Y_test = test_label
    X_test = normalize(X_test)

    X_train = normalize(X_train)  # (X_train-np.min(X_train,axis=0))/(np.max(X_train,axis=0)-np.min(X_train,axis=0))

    learning_rate = 0.001
    nn = Neuron(X_train, learning_rate=learning_rate, minbatch=80, momentum=0.9, n_hidden=200, output=1, epoch=50)
    nn.train(X_train, Y_train, X_test, Y_test)
    # print nn.predit(X_train,Y_train)


def debug():
    ########### Debug ####################
    x = np.array([[0, 0, 1],
                  [0, 1, 1],
                  [1, 0, 1],
                  [1, 1, 1]])

    y = np.array([[0],
                  [1],
                  [1],
                  [0]])
    learning_rate = 0.005
    nn = Neuron(x, learning_rate=learning_rate, minbatch=2, momentum=0.8, n_hidden=9, output=1, epoch=60)
    err = nn.train(x, y, x, y)


if __name__ == '__main__':
    """
    Create Neuron object with different input parameter and plot or output the result for
    different parameter value
    @learningrate
    @numberofHiddenlayer
    train the model and test on the test data
    @include momentum"""
    # unittest.main()
    # debug
    #debug()
    # main

    #(999)
    main()
