# -*- coding: utf-8 -*-
"""
Created on Tue Feb 09 19:41:01 2016

Nueral Network with one hidden unit layer 

@author: Tadeze
"""
import numpy as np
import random as rn
import math as mt
import matplotlib.pylab as plt

"""
Activation factory generator class
"""
class Activation(object):
    def __init__(self,activation):
        self.activation=activation
    def act(self):
        self.activations={  #list of activation functions
        'sigmoid': lambda x: 1.0/(1.0+mt.exp(x)),     
        'relu':lambda x: 1 if x>0 else 0
        }
        return self.activations[self.activation]
        
    def dact(self):
        self.diff_activation={
        #List of differentiation of the activation function
        'sigmoid': lambda x: self.activations[self.activation](x)*(1-self.activations[self.activation](x)),
        'relu': lambda x: rn.uniform(0.0,1.0) if x==0 else self.activations[self.activation](x)
                }
        return self.diff_activation[self.activation]

#define constant
N_CLASS = 2
N_DIMENSION=4
OUTPUT_NODE = 1
class Neuron(object):
    """
    include all modular size 
    """
    def __init__(self,etha,n_hidden,minbatch_size,activation_name="relu"):
        self.learning_rate=etha
        self.n_hidden = n_hidden
        self.minbatch = minbatch_size
        self.act = Activation(activation_name)
        self.softmax = Activation("sigmoid")
    def initialize_weight(self):
        self.w1 = np.ones([self.n_hidden,N_DIMENSION])
        self.w2 = np.ones(self.n_hidden,N_CLASS)

    def train(self,X_train,Y_train):


        #initialize weights
        self.initialize_weight()
        self.train_set = np.shape(X_train)[2]
        #compute first layer
        b1 = np.ones([self.train_set,self.n_hidden]) #bias in first layer
        l1= np.transpose(self.w1)*X_train +b

        #compute second layer
        b2 =np.ones([OUTPUT_NODE,self.train_set]) # bias in second layer
        l2 = np.transpose(self.w2) * self.act()(l1) +b2

        #compute output layer based on softmax
        z = self.softmax.act()(l2)  #output score for class 1

        def backprop():
            #common intermidiate term
            int_term = lambda y,z2 : (y*(1-z2) - (1-y)*z2)
            #gradient of error with w1
            dldw1  = int_term(y,z)*self.act.act()(l1)

            #gradient of error with w2
            dldw2 = int_term(y,z)*np.transpose(self.w2)*self.act.dact()(l1)*X_train











    
   def test(self,X_test,Y_test):

        

  
  #Activation functions     



#Load data
def unpicklet(file):
    import cPickle
    f = open(file,'rb')
    dict = cPickle.load(file)
    f.close()
    return dict
if __name__=='__main__':
    """
    Create Neuron object with different input parameter and plot or output the result for 
    different parameter value 
    @learningrate 
    @numberofHiddenlayer
    @minibatch size
    @activation default relu
    train the model and test on the test data 
    @include momentu
    """
