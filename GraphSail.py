# -- coding:UTF-8
import torch
# print(torch.__version__)
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [1]))

CUDA_VISIBLE_DEVICES = 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# torch.cuda.is_available() 
# torch.cuda.device_count()  
# torch.cuda.current_evice()

print(torch.cuda.get_device_name(CUDA_VISIBLE_DEVICES))

import torchvision.transforms as transforms
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn.functional as F
import torch.autograd as autograd

import pdb
from collections import defaultdict
import time
import collections
# import data_utils 
# import evaluate
from shutil import copyfile

from evaluate import *
from data_utils import *

dataset_base_path='/data/fan_xin/Gowalla'

epoch_num=100

# Gowalla
user_num=46490
item_num=57445

# Yelp
# user_num=9923
# item_num=18909

# Amazon
# user_num=10015
# item_num=12603

factor_num=256 
batch_size=1024*4
learning_rate=0.001

num_negative_test_val=-1##all

run_id="s3_sail"
print(run_id)
dataset='Gowalla'

path_save_model_base='/data/fan_xin/newlossModel_mini/'+dataset+'/s'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)

base = read(dataset_base_path + "/check_in.json", [0, 0.6])
block = read(dataset_base_path + "/check_in.json", [0.8, 0.9])
# p = propose_p(base, block)
# samples = propose_sample(base, p, 0.2) + block
training_user_set, training_item_set = list_to_set(block)
training_set_count = count_interaction(training_user_set)
user_rating_set_all = json_to_set(dataset_base_path + "/check_in.json", single=1)

print(training_set_count)

training_user_set[user_num-1].add(item_num-1)
training_item_set[item_num-1].add(user_num-1)

u_d=readD(training_user_set,user_num)
i_d=readD(training_item_set,item_num)

sparse_u_i=readTrainSparseMatrix(training_user_set,u_d,i_d,True)
sparse_i_u=readTrainSparseMatrix(training_item_set,u_d,i_d,False)

train_dataset = BPRData(
        train_dict=training_user_set, num_item=item_num, num_ng=5, is_training=True,\
        data_set_count=training_set_count, all_rating=user_rating_set_all)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=0)

model = BPR(user_num, item_num, factor_num, sparse_u_i, sparse_i_u)
model = model.to('cuda')

optimizer_bpr = torch.optim.Adam(model.parameters(), lr=learning_rate)#, betas=(0.5, 0.99))

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer_bpr, step_size=100, gamma=0.1)

PATH_model='/data/fan_xin/newlossModel_mini/'+dataset+'/s'+'s2_sail'+'/epoch'+str(7*4)+'.pt'
model.load_state_dict(torch.load(PATH_model))

old_U = model.embed_user.weight
old_I = model.embed_item.weight

n_U,n_I = loss_self(base,block,user_num,item_num)
n_U = torch.tensor(n_U).cuda()
n_I = torch.tensor(n_I).cuda()

########################### TRAINING #####################################

# testing_loader_loss.dataset.ng_sample()

print('--------training processing-------')
count, best_hr = 0, 0
for epoch in range(epoch_num):

    model.train() 
    start_time = time.time()
    train_loader.dataset.ng_sample()
    # pdb.set_trace()
    print('train data of ng_sample is  end')
    # elapsed_time = time.time() - start_time
    # print(' time:'+str(round(elapsed_time,1)))
    # start_time = time.time()

    train_loss_sum=[]
    train_loss_sum_=[]
    for user, item_i, item_j in train_loader:

        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()

        model.zero_grad()
        prediction_i, prediction_j,loss,loss_ = model(user, item_i, item_j, old_U, old_I, n_U, n_I)
        loss.backward()
        optimizer_bpr.step()
        count += 1  
        train_loss_sum.append(loss.item())  
        train_loss_sum_.append(loss_.item())

    elapsed_time = time.time() - start_time
    train_loss=round(np.mean(train_loss_sum[:-1]),4)#最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    train_loss_=round(np.mean(train_loss_sum_[:-1]),4)#最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+'\t train loss:'+str(train_loss)+"="+str(train_loss_)
    # print('--train--',elapsed_time)
    print(str_print_train)

    PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    torch.save(model.state_dict(), PATH_model)

