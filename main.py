import os
import math
from model import *
from load_data import load_data
from batch import batcher

if __name__ == "__main__":
    dirpath = os.path.dirname(__file__)
    #parameter
    vocasize = 195887
    epoch = 100
    n_in = vocasize
    n_out = 1
    lr = 0.005
    minibatch_size = 32

    train_x,train_y,dev_x,dev_y,test_x,test_y = load_data(dirpath,vocasize)
    m = math.ceil(len(train_y)/minibatch_size)
    #train
    model = LogisticRegression(n_in,n_out,lr,minibatch_size)
    print('Start Training..')
    for ep in range(epoch):
        loss = 0
        accuracy = 0
        for input_x,label_y in batcher(train_x,train_y,minibatch_size):
            y = model.forward(input_x)
            loss += model.cross_entropy_function(y,label_y)
            accuracy += model.accuracy(y,label_y)
            delta_w,delta_b = model.backward(input_x,y,label_y)
            model.update(delta_w,delta_b)
        print("Train | Epoch:{0} | Data Size:{1} | Loss:{2:.3f} | Accuracy:{3:.2f}".format(ep+1,len(train_x),loss/m,accuracy/m))
        if ep%10==0:
            y = model.forward(dev_x)
            loss = model.cross_entropy_function(y,dev_y)
            accuracy = model.accuracy(y,dev_y)
            print("Dev | Epoch:{0} | Data Size:{1} | Loss:{2:.3f} | Accuracy:{3:.2f}".format(ep+1,len(dev_y),loss,accuracy))

    #test
    y = model.forward(test_x)
    loss = model.cross_entropy_function(y,test_y)
    accuracy = model.accuracy(y,test_y)
    print("Test | Data Size:{1} | Loss:{2:.3f} | Accuracy:{3:.2f}".format(len(test_y),loss,accuracy))
