# -- coding:UTF-8 
import torch
# print(torch.__version__) 
import torch.nn as nn 

import argparse
import os
import numpy as np
import math
import sys

# os.environ["CUDA_VISIBLE_DEVICES"] =','.join(map(str, [3]))

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
# 13689
item_num=17165
# 17165
factor_num=64
batch_size=2048
top_k=20
num_negative_test_val=-1##all 

start_i_test=300
end_i_test=500
setp=10

run_id="s0"
print(run_id)
dataset='amazon-book'

path_save_model_base='./newlossModel/'+dataset+'/s'+run_id
if (os.path.exists(path_save_model_base)):
    print('has model save path')
else:
    pdb.set_trace() 


training_user_set = np.load(dataset_base_path+'/training_user_set.npy',allow_pickle=True).item()
training_item_set = np.load(dataset_base_path+'/training_item_set.npy',allow_pickle=True).item()
testing_user_set = np.load(dataset_base_path+'/testing_user_set.npy',allow_pickle=True).item()
training_set_count = 432235
user_rating_set_all = np.load(dataset_base_path+'/user_rating_set_all.npy',allow_pickle=True).item()

# mini_user_set_1 = np.load(dataset_base_path+'/mini_user_set_1.npy',allow_pickle=True).item()
# mini_item_set_1 = np.load(dataset_base_path+'/mini_item_set_1.npy',allow_pickle=True).item()
# mini_user_set_2 = np.load(dataset_base_path+'/mini_user_set_2.npy',allow_pickle=True).item()
# mini_item_set_2 = np.load(dataset_base_path+'/mini_item_set_2.npy',allow_pickle=True).item()
# mini_user_set_3 = np.load(dataset_base_path+'/mini_user_set_3.npy',allow_pickle=True).item()
# mini_item_set_3 = np.load(dataset_base_path+'/mini_item_set_3.npy',allow_pickle=True).item()
# mini_user_set_4 = np.load(dataset_base_path+'/mini_user_set_4.npy',allow_pickle=True).item()
# mini_item_set_4 = np.load(dataset_base_path+'/mini_item_set_4.npy',allow_pickle=True).item()
# mini_user_set_5 = np.load(dataset_base_path+'/mini_user_set_5.npy',allow_pickle=True).item()
# mini_item_set_5 = np.load(dataset_base_path+'/mini_item_set_5.npy',allow_pickle=True).item()

# training_user_set, training_item_set = integrateGraph(training_user_set, mini_user_set_1)
# training_user_set, training_item_set = integrateGraph(training_user_set, mini_user_set_2)
# training_user_set, training_item_set = integrateGraph(training_user_set, mini_user_set_3)
# training_user_set, training_item_set = integrateGraph(training_user_set, mini_user_set_4)
# training_user_set, training_item_set = integrateGraph(training_user_set, mini_user_set_5)


def readD(set_matrix, num_):
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

#user-item  to user-item matrix and item-user matrix

# for i in range(len(d_i_train)):
#     d_i_train[i]=[d_i_train[i]]
# for i in range(len(d_j_train)):
#     d_j_train[i]=[d_j_train[i]]

# d_i_train=torch.cuda.FloatTensor(d_i_train)
# d_j_train=torch.cuda.FloatTensor(d_j_train)
# d_i_train=d_i_train.expand(-1,factor_num)
# d_j_train=d_j_train.expand(-1,factor_num)


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

    # self.d_i_train=d_i_train
    # self.d_j_train=d_j_train 

  def forward(self):    

    users_embedding=self.embed_user.weight
    items_embedding=self.embed_item.weight

    # np.save("./users_embedding.npy", users_embedding.cpu().detach().numpy(), allow_pickle=True)
    # np.save("./items_embedding.npy", items_embedding.cpu().detach().numpy(), allow_pickle=True)

    gcn1_users_embedding = torch.sparse.mm(self.user_item_matrix, items_embedding) #+ users_embedding.mul(self.d_i_train))#*2. #+ users_embedding
    gcn1_items_embedding = torch.sparse.mm(self.item_user_matrix, users_embedding) #+ items_embedding.mul(self.d_j_train))#*2. #+ items_embedding

    gcn2_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn1_items_embedding) #+ gcn1_users_embedding.mul(self.d_i_train))#*2. + users_embedding
    gcn2_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn1_users_embedding) #+ gcn1_items_embedding.mul(self.d_j_train))#*2. + items_embedding
      
    gcn3_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn2_items_embedding) #+ gcn2_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
    gcn3_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn2_users_embedding) #+ gcn2_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding

    # gcn4_users_embedding = torch.sparse.mm(self.user_item_matrix, gcn3_items_embedding) #+ gcn3_users_embedding.mul(self.d_i_train))#*2. + gcn1_users_embedding
    # gcn4_items_embedding = torch.sparse.mm(self.item_user_matrix, gcn3_users_embedding) #+ gcn3_items_embedding.mul(self.d_j_train))#*2. + gcn1_items_embedding
    
    # gcn_users_embedding = torch.cat((users_embedding,gcn1_users_embedding,gcn2_users_embedding,gcn3_users_embedding,gcn4_users_embedding),-1)#+gcn4_users_embedding
    # gcn_items_embedding = torch.cat((items_embedding,gcn1_items_embedding,gcn2_items_embedding,gcn3_items_embedding,gcn4_items_embedding),-1)#+gcn4_items_embedding#

    gcn_users_embedding = users_embedding + (1/2)*gcn1_users_embedding + (1/3)*gcn2_users_embedding + (1/4)*gcn3_users_embedding
    gcn_items_embedding = items_embedding + (1/2)*gcn1_items_embedding + (1/3)*gcn2_items_embedding + (1/4)*gcn3_items_embedding

    return gcn_users_embedding, gcn_items_embedding

model = BPR(user_num, item_num, factor_num,sparse_u_i,sparse_i_u)
model=model.to('cuda')

optimizer_bpr = torch.optim.Adam(model.parameters(), lr=0.001)#, betas=(0.5, 0.99))

test_batch=52#int(batch_size/32)
testing_dataset = resData(train_dict=testing_user_set, batch_size=test_batch,num_item=item_num,all_pos=training_user_set)
testing_loader = DataLoader(testing_dataset,batch_size=1, shuffle=False, num_workers=0)

########################### TRAINING ##################################### 
# testing_loader_loss.dataset.ng_sample() 

print('--------test processing-------') 
count, best_hr = 0, 0
for epoch in range(start_i_test,end_i_test,setp):
    model.train()   
    
    embed_user = torch.tensor(np.load(path_save_model_base+'/U_epoch'+str(epoch)+'.npy', allow_pickle=True)).cuda()
    embed_item = torch.tensor(np.load(path_save_model_base+'/I_epoch'+str(epoch)+'.npy', allow_pickle=True)).cuda()
    
    model.embed_user.weight = torch.nn.Parameter(embed_user)
    model.embed_item.weight = torch.nn.Parameter(embed_item)
    
#     PATH_model=path_save_model_base+'/epoch'+str(epoch)+'.pt'
#     #torch.save(model.state_dict(), PATH_model)
#     model.load_state_dict(torch.load(PATH_model))

    model.eval()
    # ######test and val###########    
    
    gcn_users_embedding, gcn_items_embedding = model()
    
    user_e=gcn_users_embedding.cpu().detach().numpy()
    item_e=gcn_items_embedding.cpu().detach().numpy()
    all_pre=np.matmul(user_e,item_e.T)
    
    HR, NDCG = [], []
    set_all=set(range(item_num))  
    #spend 461s 
    test_start_time = time.time()
    for u_i in testing_user_set:
        
        item_i_list = list(testing_user_set[u_i])
        index_end_i = len(item_i_list)
        item_j_list = list(set_all-training_user_set[u_i]-testing_user_set[u_i])
        item_i_list.extend(item_j_list)

#         pre_one = np.matmul(user_e[u_i:u_i+1], item_e[item_i_list].T)[0]
        pre_one=all_pre[u_i][item_i_list]
        
        indices=largest_indices(pre_one, top_k)
        indices=list(indices[0])
        
        hr_t,ndcg_t=hr_ndcg(indices,index_end_i,top_k)
        elapsed_time = time.time() - test_start_time
        HR.append(hr_t)
        NDCG.append(ndcg_t)
    
    hr_test=round(np.mean(HR),4)
    ndcg_test=round(np.mean(NDCG),4)    
    
    # test_loss,hr_test,ndcg_test = evaluate.metrics(model,testing_loader,top_k,num_negative_test_val,batch_size)  
    str_print_evl="epoch:"+str(epoch)+'time:'+str(round(elapsed_time,2))+"\t test"+" hit:"+str(hr_test)+' ndcg:'+str(ndcg_test) 
    print(str_print_evl)

