import os
import numpy as np

class LogisticRegression(object):
    def __init__(self, n_in, n_out, lr, minibatch_size):
        self.W = np.zeros((n_in,1))
        self.b = np.zeros(n_out)
        self.learn_rate = lr
        self.minibatch_size = minibatch_size

    def linear(self,input_x):
        return np.dot(input_x,self.W)+self.b

    def sigmoid(self,input_x):
        return 1 / (1 + np.exp(-input_x))

    def forward(self,input_x):
        return self.sigmoid(self.linear(input_x)) >= 0.5

    def backward(self,input_x,label_y):
        pre_y = self.sigmoid(self.linear(input_x))
        delta_w = np.dot(input_x.T,(pre_y - label_y))/len(label_y)
        delta_b = np.sum(pre_y - label_y)/len(label_y)
        return delta_w,delta_b

    def update(self,input_x,label_y):
        delta_w,delta_b = self.backward(input_x,label_y)
        self.W -= delta_w*self.learn_rate
        self.b -= delta_b*self.learn_rate


    def cross_entropy_function(self,input_x,label_y):
        e=0.00001
        pre_y = self.sigmoid(self.linear(input_x))
        cross_entropy_loss = - np.mean(np.sum(label_y * np.log(pre_y+e) +(1 - label_y) * np.log(1 - pre_y+e),axis=1))
        return cross_entropy_loss

    def accuracy(self,input_x,label_y):
        pre_y = self.forward(input_x)
        label_y = label_y >= 1
        return np.sum(pre_y==label_y)/len(label_y)
