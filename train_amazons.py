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

dataset_base_path='./Amazon' 

user_num=13689
item_num=17165
factor_num=64
batch_size=2048
top_k=5
num_negative_test_val=-1##all 

run_id="s0"
print(run_id)
dataset='amazon-book'

path_save_model_base='./newlossModel/'+dataset+'/s'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    os.makedirs(path_save_model_base) 

training_user_set = np.load(dataset_base_path+'/training_user_set.npy',allow_pickle=True).item()
training_item_set = np.load(dataset_base_path+'/training_item_set.npy',allow_pickle=True).item()
training_set_count = 432235
user_rating_set_all = np.load(dataset_base_path+'/user_rating_set_all.npy',allow_pickle=True).item()

# training_user_set = np.load(dataset_base_path+'/mini_user_set_1.npy',allow_pickle=True).item()
# training_item_set = np.load(dataset_base_path+'/mini_item_set_1.npy',allow_pickle=True).item()
# training_set_count = 1377
# user_rating_set_all = np.load(dataset_base_path+'/user_rating_set_all.npy',allow_pickle=True).item()

# maxU = -1
# maxI = -1

# for k,v in user_rating_set_all.items():

# 	if k > maxU:
# 		maxU = k
# 	for i in v:
# 		if i > maxI:
# 			maxI = i

# print(maxU, maxI)


# max_user 162113 {24656, 100321, 3357}
# max_item 107980 {160828, 130846}

# training_user_set[162113].add(107980)
# training_item_set[107980].add(162113)

def readD(set_matrix,num_):
    user_d=[]
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)  
        user_d.append(len_set)
    return user_d
u_d=readD(training_user_set,user_num)
i_d=readD(training_item_set,item_num)
#1/(d_i+1)
d_i_train=u_d
d_j_train=i_d
#1/sqrt((d_i+1)(d_j+1))
# d_i_j_train=np.sqrt(u_d*i_d)


#user-item  to user-item matrix and item-user matrix
def readTrainSparseMatrix(set_matrix,is_user):
    user_items_matrix_i=[]
    user_items_matrix_v=[]
    if is_user:
        d_i=u_d
        d_j=i_d
    else:
        d_i=i_d
        d_j=u_d
    for i in set_matrix:
        len_set=len(set_matrix[i])  
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            d_i_j=np.sqrt(d_i[i]*d_j[j])
            #1/sqrt((d_i+1)(d_j+1))
            user_items_matrix_v.append(d_i_j)#(1./len_set) 

    user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)

sparse_u_i=readTrainSparseMatrix(training_user_set,True)
sparse_i_u=readTrainSparseMatrix(training_item_set,False)

# pdb.set_trace()

class BPR(nn.Module):
    def __init__(self, user_num, item_num, factor_num,user_item_matrix,item_user_matrix):
        super(BPR, self).__init__()
        """
        user_num: number of users;
        item_num: number of items;
        factor_num: number of predictive factors.
        """
        self.user_item_matrix = user_item_matrix
        self.item_user_matrix = item_user_matrix
        self.embed_user = nn.Embedding(user_num, factor_num)
        self.embed_item = nn.Embedding(item_num, factor_num)

        nn.init.normal_(self.embed_user.weight, std=0.01)
        nn.init.normal_(self.embed_item.weight, std=0.01)

    def forward(self, user, item_i, item_j):    
        
        users_embedding=self.embed_user.weight
        items_embedding=self.embed_item.weight

        gcn1_users_embedding = torch.sparse.mm(self.user_item_matrix, items_embedding) #+ users_embedding.mul(self.d_i_train))#*2. #+ users_embedding
        gcn1_items_embedding = torch.sparse.mm(self.item_user_matrix, users_embedding) #+ items_embedding.mul(self.d_j_train))#*2. #+ items_embedding
        
        gcn2_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) #+ gcn1_users_embedding.mul(self.d_i_train))#*2. + users_embedding
        gcn2_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) #+ gcn1_items_embedding.mul(self.d_j_train))#*2. + items_embedding
          
        gcn3_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) #+ gcn2_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
        gcn3_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) #+ gcn2_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding
        
        gcn_users_embedding = users_embedding + (1/2)*gcn1_users_embedding + (1/3)*gcn2_users_embedding + (1/4)*gcn3_users_embedding
        gcn_items_embedding = items_embedding + (1/2)*gcn1_items_embedding + (1/3)*gcn2_items_embedding + (1/4)*gcn3_items_embedding
        
        user = F.embedding(user,gcn_users_embedding)
        item_i = F.embedding(item_i,gcn_items_embedding)
        item_j = F.embedding(item_j,gcn_items_embedding)  
        # # pdb.set_trace()
        prediction_i = (user * item_i).sum(dim=-1)
        prediction_j = (user * item_j).sum(dim=-1) 
        # loss=-((rediction_i-prediction_j).sigmoid())**2#self.loss(prediction_i,prediction_j)#.sum()
        l2_regulization = 0.001*(user**2+item_i**2+item_j**2).sum(dim=-1)
        # l2_regulization = 0.01*((gcn1_users_embedding**2).sum(dim=-1).mean()+(gcn1_items_embedding**2).sum(dim=-1).mean())
        
        loss2= -((prediction_i - prediction_j).sigmoid().log().mean())
        # loss= loss2 + l2_regulization
        loss= -((prediction_i - prediction_j)).sigmoid().log().mean() + l2_regulization.mean()
        # pdb.set_trace()
        return prediction_i, prediction_j, loss, loss2

train_dataset = BPRData(
        train_dict=training_user_set, num_item=item_num, num_ng=5, is_training=True,\
        data_set_count=training_set_count,all_rating=user_rating_set_all)
train_loader = DataLoader(train_dataset,
        batch_size=batch_size, shuffle=True, num_workers=0)

model = BPR(user_num, item_num, factor_num, sparse_u_i,sparse_i_u)

model=model.to('cuda')

# embed_user = torch.tensor(np.load("./Parameters_0/gcn_users_embedding" + str(49) + ".npy", allow_pickle=True)).cuda()
# embed_item = torch.tensor(np.load("./Parameters_0/gcn_items_embedding" + str(49) + ".npy", allow_pickle=True)).cuda()

# model.embed_user.weight = torch.nn.Parameter(embed_user)
# model.embed_item.weight = torch.nn.Parameter(embed_item)

optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.001)#, betas=(0.5, 0.99))


########################### TRAINING #####################################

# testing_loader_loss.dataset.ng_sample() 

print('--------training processing-------')
count, best_hr = 0, 0
for epoch in range(500):
    model.train() 
    start_time = time.time()
    train_loader.dataset.ng_sample()
    # pdb.set_trace()
    print('train data of ng_sample is  end')
    # elapsed_time = time.time() - start_time
    # print(' time:'+str(round(elapsed_time,1)))
    # start_time = time.time()
    
    train_loss_sum=[]
    train_loss_sum2=[]
    for user, item_i, item_j in train_loader:
        user = user.cuda()
        item_i = item_i.cuda()
        item_j = item_j.cuda()
        
        model.zero_grad()
        prediction_i, prediction_j,loss,loss2 = model(user, item_i, item_j)
        loss.backward()
        optimizer_bpr.step() 
        count += 1  
        train_loss_sum.append(loss.item())  
        train_loss_sum2.append(loss2.item())  
        # print(count)

    elapsed_time = time.time() - start_time
    train_loss=round(np.mean(train_loss_sum[:-1]),4)#最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    train_loss2=round(np.mean(train_loss_sum2[:-1]),4)#最后一个可能不满足一个batch，所以去掉这样loss就是一致的可以求mean了
    str_print_train="epoch:"+str(epoch)+' time:'+str(round(elapsed_time,1))+'\t train loss:'+str(train_loss)+"="+str(train_loss2)+"+" 
    print('--train--',elapsed_time)
    print(str_print_train)

    # PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
    # torch.save(model.state_dict(), PATH_model)
    
    np.save(path_save_model_base+'/U_epoch'+str(epoch)+'.npy', model.embed_user.weight.cpu().detach().numpy(), allow_pickle=True)
    np.save(path_save_model_base+'/I_epoch'+str(epoch)+'.npy', model.embed_item.weight.cpu().detach().numpy(), allow_pickle=True)

