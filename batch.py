import numpy as np
import random

def batcher(data_x,data_y,minibatch_size,shuffl=True):
    minbatch_x = np.empty((0,len(data_x[0])))
    minbatch_y = np.empty((0,len(data_y[0])))
    index_id = list(range(len(data_x)))
    if shuffl:
        random.shuffle(index_id)
    for i,id in enumerate(index_id):
        minbatch_x = np.append(minbatch_x,data_x[id,:].reshape(1,len(data_x[id,:])),axis=0)
        minbatch_y = np.append(minbatch_y,data_y[id,:].reshape(1,len(data_y[id,:])),axis=0)
        if len(minbatch_x)==minibatch_size or i==len(data_x)-1:
            yield minbatch_x,minbatch_y
            minbatch_x = np.empty((0,len(data_x[0])))
            minbatch_y = np.empty((0,len(data_y[0])))
