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
from Constant.py import *

epoch_num=100

factor_num=256 
batch_size=1024*4
learning_rate=0.0005

alpha = 0.05

num_negative_test_val=-1##all

run_id=RUN_ID
print(run_id)

if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base)

base = read(dataset_base_path + "/check_in.json", [0, 0.6])
block = read(dataset_base_path + "/check_in.json", [0.6, 0.7])
p = propose_p(base, block)
samples = propose_sample(base, p, alpha) + block
training_user_set, training_item_set = list_to_set(samples)
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

# embed_user = torch.tensor(np.load("./Parameters_0/gcn_users_embedding" + str(49) + ".npy", allow_pickle=True)).cuda()
# embed_item = torch.tensor(np.load("./Parameters_0/gcn_items_embedding" + str(49) + ".npy", allow_pickle=True)).cuda()

# model.embed_user.weight = torch.nn.Parameter(embed_user)
# model.embed_item.weight = torch.nn.Parameter(embed_item)

optimizer_bpr = torch.optim.Adam(model.parameters(), lr=learning_rate)#, betas=(0.5, 0.99))

# scheduler = torch.optim.lr_scheduler.StepLR(optimizer_bpr, step_size=100, gamma=0.1)

PATH_model='/data/fan_xin/newlossModel_mini/'+dataset+'/s'+'s0_full_batch_0.001'+'/epoch'+str(499)+'.pt'
model.load_state_dict(torch.load(PATH_model))

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
        prediction_i, prediction_j,loss,loss_ = model(user, item_i, item_j)
        loss.backward()
        optimizer_bpr.step()
        count += 1  
        train_loss_sum.append(loss.item())  
        train_loss_sum_.append(loss_.item())
        # print(count)

    # scheduler.step()

    elapsed_time = time.time() - start_time
    train_loss=round(np.mean(train_loss_sum[:-1]),4)#最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    train_loss_=round(np.mean(train_loss_sum_[:-1]),4)#最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+'\t train loss:'+str(train_loss)+"="+str(train_loss_)
    # print('--train--',elapsed_time)
    print(str_print_train)

    PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    torch.save(model.state_dict(), PATH_model)

    # change the graph structure

    samples = propose_sample(base, p, alpha) + block
    training_user_set, training_item_set = list_to_set(samples)

    training_set_count = count_interaction(training_user_set)

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

    model.user_item_matrix = sparse_u_i
    model.item_user_matrix = sparse_i_u

    # np.save(path_save_model_base+'/U_epoch'+str(epoch)+'.npy', model.embed_user.weight.cpu().detach().numpy(), allow_pickle=True)
    # np.save(path_save_model_base+'/I_epoch'+str(epoch)+'.npy', model.embed_item.weight.cpu().detach().numpy(), allow_pickle=True)