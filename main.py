import os
from model import *
from load_data import load_data
from batch import batcher

def evaluate(LG,input_x,label_y):
    Loss = LG.cross_entropy_function(input_x,label_y)
    Accuracy = LG.accuracy(input_x,label_y)
    return Loss,Accuracy

if __name__ == "__main__":
    dirpath = os.path.dirname(__file__)
    #parameter
    vocasize = 195887
    epoch = 40
    n_in = vocasize
    n_out = 1
    lr = 0.005
    minibatch_size = 32

    train_x,train_y,dev_x,dev_y,test_x,test_y = load_data(dirpath,vocasize)

    #train
    LG = LogisticRegression(n_in,n_out,lr,minibatch_size)
    print('Start Training..')
    for ep in range(epoch):
        for input_x,label_y in batcher(train_x,train_y,minibatch_size):
            Loss,Accuracy = evaluate(LG,input_x,label_y)
            LG.update(input_x,label_y)
            print("Train | Epoch:{0} | Data Size:{1} | Loss:{2:.3f} | Accuracy:{3:.2f}%".format(ep+1,len(label_y),Loss,Accuracy*100))
        Loss,Accuracy = evaluate(LG,dev_x,dev_y)
        print("Dev | Epoch:{0} | Data Size:{1} | Loss:{2:.3f} | Accuracy:{3:.2f}%".format(ep+1,len(dev_y),Loss,Accuracy*100))

    #test
    Loss,Accuracy = evaluate(LG,test_x,test_y)
    print("Test | Data Size:{1} | Loss:{2:.3f} | Accuracy:{3:.2f}%".format(ep+1,len(test_y),Loss,Accuracy*100))
