# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 19:41:01 2016

Neural Network with one hidden unit layer

@author: Tadesse Zemicheal
"""
import numpy as np

"""
Activation factory generator class
"""
class Activation(object):
    def __init__(self,activation):
        self.activation = activation

    def act(self):
        self.activations = {  # list of activation functions
            'sigmoid': lambda x: self.sigmoid(x),
            'relu': lambda x: 0 if x<0 else x
        }
        return self.activations[self.activation]
    def dact(self):
        self.diff_activation = {
            # List of differentiation of the activation function
            'sigmoid': lambda x: self.sigmoid(x) * (1 - self.sigmoid(x)),
            'relu': lambda x: self.d_relu(x)
        }
        return self.diff_activation[self.activation]


# Activation function body
    def sigmoid(self,x):
        if x > 35:
            x = 35
        if x < -35:
            x = -35
        return 1.0 / (1.0 + np.exp(-x))

    def d_relu(self,x):
        if x > 0:
            return 1
        elif x == 0:
            return np.random.uniform(0, 1)
        else:
            return 0

"""
Main Neural network layer
"""
class Neuron(object):
    def __init__(self, X, learning_rate, minbatch, momentum, n_hidden, output, epoch,activation="relu"):
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
        self.activation =Activation(activation)
        self.softmax = Activation("sigmoid")
    """ Compute entropy loss
    """
    def evaluate_loss(self, score, pred):
        yl = len(pred)
        score=np.clip(score,1e-3,1-1e-3)  #clip for overflow of 1 or 0
        pred=np.clip(pred.T,1e-3,1-1e-3)
        return  -np.sum((score * np.log(pred) + (1 - score) * np.log(1 - pred)))

    """
    Non-linear activation function
    """
    def nonlinear(self,value,deriv=False):
        if deriv:
            return np.vectorize(self.activation.dact(), otypes=[np.float64])(value)
        else:
            return np.vectorize(self.activation.act(), otypes=[np.float64])(value)

    """
    Forward pass of a training examples DxN dimension
     returns Nx1 score, l1 layer and l2 layer value
    """
    def forward(self, x_train):
        l1 = np.dot(self.w1, x_train.T) + self.b1  #layer 1
        nonlinear_l1 =self.nonlinear(l1)  # hidden layer
        l2 = np.dot(self.w2, nonlinear_l1) + self.b2  #hidden -to -outpu
        z = np.vectorize(self.softmax.act())(l2)  #softmax
        return z, l1, l2

    """
    Backprop error to layers
    """
    def backprop(self, X, Y,Z, l1, l2):
        N = len(Y)
        # error from output layer
        l2_error = Z-Y
        # gradient of output layer
        nonl_l1 =self.nonlinear(l1)
        dldw2 = nonl_l1.dot(l2_error).T /N  # gradient of w2

        # error from hidden unit layer
        l1_error = l2_error.dot(self.w2)
        dldw1_delta = l1_error.T * self.nonlinear(l1,deriv=True)
        dldw1 = dldw1_delta.dot(X)/N #gradient of input layer

        #gradient of bias
        grad_b1 = np.mean(dldw1_delta, axis=1, keepdims=True)
        grad_b2 = np.mean(l2_error,keepdims=True)

        return dldw1, dldw2, grad_b1, grad_b2

    #compute accuray and entropy loss prediction
    def predict(self, x_test, y_test):
        z, l1, l2 = self.forward(x_test)   #move forward to prediction
        accuray = 1 - np.mean(np.abs(np.round(z) - y_test.T))
        loss = self.evaluate_loss(y_test, z)
        return accuray, loss/len(y_test)

    #Train Network and check accuracy of testing in each epoch
    def train(self, x_train, y_train, test_data, test_label):

        filename = 'output_hidden'+str(self.n_hidden)+'_lrate_'+str(self.learning_rate)+'_minbatch_'+str(self.minbatch)+"_epoch_"+str(self.epoch)+".csv"
        #log = open(filename, 'w')

        train_size, dimension = np.shape(x_train)
        self.momentum_w1 = np.zeros_like(self.w1)
        self.momentum_w2 = np.zeros_like(self.w2)
        self.momentum_b1 = np.zeros_like(self.b1)
        self.momentum_b2 = np.zeros_like(self.b2)

        # Number of epoch to run

        for ep in xrange(self.epoch):
            # sample stochastic minibatch gradient descent
            self.SGD(train_size, x_train, y_train) #check prediction in every epoch
        #training_err = self.predict(x_train, y_train)
        #testing_err = self.predict(test_data, test_label)

        #print (training_err, testing_err)
        #print "\n"
        #log.write( str(ep) + "," + str(training_err[0]) + "," +str(training_err[1])+',' +str(testing_err[0])+','+str(testing_err[1]))
        #log.write("\n")

        #log.close()
        print "Iteration completed"

    def SGD(self, train_size, x_train, y_train):
        n_sample = range(train_size)
        np.random.shuffle(n_sample)
        # random shuffle in every epch
        it = 0
        while it < train_size - self.minbatch:
            # Forward pass for all batch size  and compute error
            # update w1 and w2 based on the error size
            # for i in range(it,it+self.minbatch-1):

            x = x_train[it:self.minbatch + it, ]  # slice on batch of training examples
            y = y_train[it:self.minbatch + it]

            z, l1, l2 = self.forward(x)  # forward pass
            dw1, dw2, grad_b1, grad_b2 = self.backprop(x, y,z.T,l1, l2)  # backprop batch examples

            # update momentum
            self.momentum_w1 = self.momentum * self.momentum_w1 - self.learning_rate * dw1
            self.momentum_w2 = self.momentum * self.momentum_w2 - self.learning_rate * dw2

            self.momentum_b1 = self.momentum * self.momentum_b1 + self.learning_rate * grad_b1
            self.momentum_b2 = self.momentum * self.momentum_b2 + self.learning_rate * grad_b2

            # update weight
            self.w2 = self.w2 + self.momentum_w2
            self.w1 = self.w1 + self.momentum_w1
            self.b1 = self.b1 + self.momentum_b1
            self.b2 = self.b2 + self.momentum_b2

            it += self.minbatch


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

    Y_test = test_label
    Y_train = train_label
    X_test = normalize(test_data)
    X_train = normalize(train_data)

    ## -------- Run experiment --------------
    nn = Neuron(X_train, learning_rate=0.001, minbatch=50, momentum=0.9, n_hidden=50, output=1, epoch=50)
    nn.train(X_train, Y_train, X_test, Y_test)
    accuracy=nn.predict(X_test,Y_test)


    """
     Uncomment to run experiment with multiple tunning parameter
    """
    ######### To experiment with tunning parameters
    # alphas = [0.001,0.0005]
    # hidden_units = [40,60,80,120]
    # minibatchs = [30,50,120,250]
    # lrate = [0.1,0.001,0.005,0.0001]
    # #learning rate
    # filename ='learningcurve.csv'
    # log =open(filename,'w')
    # for bch in hidden_units:
    #     nn = Neuron(X_train, learning_rate=0.001, minbatch=50, momentum=0.9, n_hidden=bch, output=1, epoch=50)
    #     nn.train(X_train, Y_train, X_test, Y_test)
    #     acc=nn.predict(X_test,Y_test)
    #     log.write( str(bch) + "," + str(acc[0]) + "," +str(acc[1])+'\n')
    # log.close()
    # # print nn.predit(X_train,Y_train)
if __name__ == '__main__':
    """
    """
    # unittest.main()
    # debug
    #debug()
    # main
    main()
