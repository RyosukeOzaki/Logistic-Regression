import os
import numpy as np
import random

def load_data(dirpath,vocasize):
    #init_index
    data={}
    data_x={}
    data_y={}
    #load data
    label = ["negative","positive"]
    print("Data Loading...")
    for type_set in label:
        fname = os.path.join(dirpath,"data/books/{0}.review".format(type_set))
        data[type_set] = [line.strip() for line in open(fname,"r")]
        data[type_set] = [word.split(" ") for word in data[type_set]]
        data_x[type_set] = np.zeros((len(data[type_set]),vocasize))
        data_y[type_set] = np.zeros((len(data[type_set]),1))
        for row in range(len(data[type_set])):
            for pair in data[type_set][row]:
                pair = pair.split(":")
                word_id = int(pair[0])
                freq = int(pair[1])
                data_x[type_set][row,word_id] = freq
            if type_set=="positive":
                data_y[type_set][row] = 1
            elif type_set=="negative":
                data_y[type_set][row] = 0
    train_x,train_y,dev_x,dev_y,test_x,test_y = split_data(data_x["positive"],data_y["positive"],data_x["negative"],data_y["negative"])
    print("Data Load done")
    return train_x,train_y,dev_x,dev_y,test_x,test_y

def split_data(data_pos_x,data_pos_y,data_neg_x,data_neg_y):
    i_pos_num,i_neg_num = int(0.8*len(data_pos_x)),int(0.8*len(data_neg_x))
    j_pos_num,j_neg_num = int(0.1*len(data_pos_x)),int(0.1*len(data_neg_x))

    train_x= np.append(data_pos_x[0:i_pos_num],data_neg_x[0:i_neg_num],axis=0)
    train_y= np.append(data_pos_y[0:i_pos_num],data_neg_y[0:i_neg_num],axis=0)
    dev_x= np.append(data_pos_x[i_pos_num:i_pos_num+j_pos_num],data_neg_x[i_neg_num:i_neg_num+j_neg_num],axis=0)
    dev_y= np.append(data_pos_y[i_pos_num:i_pos_num+j_pos_num],data_neg_y[i_neg_num:i_neg_num+j_neg_num],axis=0)
    test_x= np.append(data_pos_x[i_pos_num+j_pos_num::],data_neg_x[i_neg_num+j_neg_num::],axis=0)
    test_y= np.append(data_pos_y[i_pos_num+j_pos_num:],data_neg_y[i_neg_num+j_neg_num:],axis=0)

    return train_x,train_y,dev_x,dev_y,test_x,test_y
