# -- coding:UTF-8
import numpy as np 
import pandas as pd 
import scipy.sparse as sp 
import scipy

import torch.nn as nn 
import torch.utils.data as data
import pdb
from torch.autograd import Variable
import torch
import math
import random
import collections
import json

import torch.nn.functional as F

# Gowalla
user_num=46490
item_num=57445

# Yelp
# user_num=9923
# item_num=18909

# Amazon
# user_num=10015
# item_num=12603

# Random sample from list
def random_sample(All, rate):

    return np.random.choice(All, replace=False, size=round(len(All)*rate)).tolist()

# Random sample from list
def greedy_sample(All, rate):

    U, I = list_to_set(All)
    p = list()

    for line in All:
        temp_p = 1/len(U[line["user_id"]])
        temp_p = pow(temp_p, 2)
        p.append(temp_p)

    sum_p = sum(p)
    p = [x/sum_p for x in p]

    return np.random.choice(All, replace=False, size=round(len(All)*rate), p=p).tolist()

def propose_p(All, block):

    factors = list()
    user_sub, item_sub = number_subGraph(block)
    graph_map = collections.Counter()

    for i in user_sub:
        graph_map[i] += 1
    for i in item_sub:
        graph_map[i] += 1

    for line in All:
        if not user_sub[line["user_id"]] and not item_sub[line["item_id"]]:
            factors.append(1)
        if user_sub[line["user_id"]] and not item_sub[line["item_id"]]:
            factors.append(0.1)
        if not user_sub[line["user_id"]] and item_sub[line["item_id"]]:
            factors.append(0.1)
        if user_sub[line["user_id"]] and item_sub[line["item_id"]]:
            if user_sub[line["user_id"]] == item_sub[line["item_id"]]:
                factors.append(0.1)
            else:
                # factors.append( min(graph_map[user_sub[line["user_id"]]], graph_map[item_sub[line["item_id"]]]) )
                factors.append(0.1)

    return factors

# propose sample from list
def propose_sample(All, propose_p, rate):

    sum_p = sum(propose_p)
    propose_p = [x/sum_p for x in propose_p]

    return np.random.choice(All, replace=False, size=round(len(All)*rate), p=propose_p).tolist()

# Read json to list
def read(url, rate=[0, 1]):

    All = list()

    with open(url, mode='r') as f:

        line = f.readline()

        while line:
        
            line = json.loads(line)
            
            All.append(line)
            
            line = f.readline()

    f.close()

    return All[round(len(All)*rate[0]):round(len(All)*rate[1])]

# Generate Set from List
def list_to_set(All, rate=[0, 1], single=0):

    training_user_set = collections.defaultdict(set)
    training_item_set = collections.defaultdict(set)

    for line in All[round(len(All)*rate[0]):round(len(All)*rate[1])]:
        training_user_set[line["user_id"]].add(line["item_id"])
        training_item_set[line["item_id"]].add(line["user_id"])

    if single:
        return training_user_set
    else:
        return training_user_set, training_item_set

# Generate trainSet from json
def json_to_set(url, rate=[0, 1], single=0):

    training_user_set = collections.defaultdict(set)
    training_item_set = collections.defaultdict(set)

    All = list()
    
    with open(url, mode='r') as f:

        line = f.readline()

        while line:

            line = json.loads(line)

            All.append(line)
    
            line = f.readline()

    f.close()

    for line in All[round(len(All)*rate[0]):round(len(All)*rate[1])]:
        training_user_set[line['user_id']].add(line['item_id'])
        training_item_set[line['item_id']].add(line['user_id'])

    if single:
        return training_user_set
    else:
        return training_user_set, training_item_set

# Count the number of interations
def count_interaction(Set):

    count = 0
    for k,v in Set.items():
        for i in v:
            count += 1
    return count

def count_user_item(Set):

    I = set()
    for k,v in Set.items():
        for i in v:
            I.add(i)
    return len(Set), len(I)

# Calculate the number of subGraph
def number_subGraph(block):

    user_set, item_set = list_to_set(block)

    user_sub = [0]*user_num
    item_sub = [0]*item_num
    number = 0

    for k in list(user_set.keys()):
        if not user_sub[k]:
            number += 1
            start = (k,'u')
            queue = [start]
            while queue:
                node = queue.pop(0)
                if node[1] == 'u':
                    for neighbor in user_set[node[0]]:
                        if not item_sub[neighbor]:
                            item_sub[neighbor] = number
                            queue.append((neighbor, "i"))
                if node[1] == "i":
                    for neighbor in item_set[node[0]]:
                        if not user_sub[neighbor]:
                            user_sub[neighbor] = number
                            queue.append((neighbor, "u"))

    return user_sub, item_sub


def load_all(test_num=100):
    """ We load all the three file here to save time in each epoch. """
    train_data = pd.read_csv(
        '../data/ml-1m.train.rating', 
        sep='\t', header=None, names=['user', 'item'], 
        usecols=[0, 1], dtype={0: np.int32, 1: np.int32})

    user_num = train_data['user'].max() + 1
    item_num = train_data['item'].max() + 1

    train_data = train_data.values.tolist()

    # load ratings as a dok matrix
    train_mat = sp.dok_matrix((user_num, item_num), dtype=np.float32)
    for x in train_data:
        train_mat[x[0], x[1]] = 1.0

    test_data = []
    with open('../data/ml-1m.test.rating', 'r') as fd:
        line = fd.readline()
        while line != None and line != '':
            arr = line.split('\t')
            u = eval(arr[0])[0]
            test_data.append([u, eval(arr[0])[1]])
            for i in arr[1:]:
                test_data.append([u, int(i)])
            line = fd.readline()
    return train_data, test_data, user_num, item_num, train_mat

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

        # old_U=0, old_I=0, n_U=0, n_I=0

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
        l2_regulization = 0.0001*(user**2+item_i**2+item_j**2).sum(dim=-1)
        # l2_regulization = 0.01*((gcn1_users_embedding**2).sum(dim=-1).mean()+(gcn1_items_embedding**2).sum(dim=-1).mean())

        loss_ = -((prediction_i - prediction_j).sigmoid().log().mean()) + loss_self
        # loss= loss2 + l2_regulization
        loss = -((prediction_i - prediction_j)).sigmoid().log().mean() + l2_regulization.mean()
        # pdb.set_trace()
        return prediction_i, prediction_j, loss, loss_


class BPR_(nn.Module):
  def __init__(self, user_num, item_num, factor_num,user_item_matrix,item_user_matrix):
    super(BPR_, self).__init__()
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

def readD(set_matrix,num_):
    user_d=[]
    for i in range(num_):
        len_set=1.0/(len(set_matrix[i])+1)  
        user_d.append(len_set)
    return user_d

def readTrainSparseMatrix(set_matrix,u_d,i_d,is_user):
    user_items_matrix_i=[]
    user_items_matrix_v=[]
    if is_user:
        d_i=u_d
        d_j=i_d
    else:
        d_i=i_d
        d_j=u_d
    for i in set_matrix:
        # len_set=len(set_matrix[i])
        for j in set_matrix[i]:
            user_items_matrix_i.append([i,j])
            d_i_j=np.sqrt(d_i[i]*d_j[j])
            #1/sqrt((d_i+1)(d_j+1))
            user_items_matrix_v.append(d_i_j)#(1./len_set) 

    user_items_matrix_i=torch.cuda.LongTensor(user_items_matrix_i)
    user_items_matrix_v=torch.cuda.FloatTensor(user_items_matrix_v)
    return torch.sparse.FloatTensor(user_items_matrix_i.t(), user_items_matrix_v)

class BPRData(data.Dataset):
    def __init__(self,train_dict=None,num_item=0, num_ng=1, is_training=None, data_set_count=0,all_rating=None):
        super(BPRData, self).__init__()
        
        self.num_item = num_item
        self.train_dict = train_dict
        self.num_ng = num_ng
        self.is_training = is_training
        self.data_set_count = data_set_count
        self.all_rating=all_rating
        self.set_all_item=set(range(num_item))  

    def ng_sample(self):
        # assert self.is_training, 'no need to sampling when testing'
        # print('ng_sample----is----call-----') 
        self.features_fill = []
        for user_id in self.train_dict:
            positive_list=self.train_dict[user_id]#self.train_dict[user_id]
            all_positive_list=self.all_rating[user_id]
            #item_i: positive item ,,item_j:negative item   
            # temp_neg=list(self.set_all_item-all_positive_list)
            # random.shuffle(temp_neg)
            # count=0
            # for item_i in positive_list:
            #     for t in range(self.num_ng):   
            #         self.features_fill.append([user_id,item_i,temp_neg[count]])
            #         count+=1
            for item_i in positive_list:
                for t in range(self.num_ng):
                    item_j=np.random.randint(self.num_item)
                    while item_j in all_positive_list:
                        item_j=np.random.randint(self.num_item)
                    self.features_fill.append([user_id,item_i,item_j])

    def __len__(self):  
        return self.num_ng*self.data_set_count#return self.num_ng*len(self.train_dict)


    def __getitem__(self, idx):
        features = self.features_fill
        
        user = features[idx][0]
        item_i = features[idx][1]
        item_j = features[idx][2]
        return user, item_i, item_j


class resData(data.Dataset):
    def __init__(self,train_dict=None,batch_size=0,num_item=0,all_pos=None):
        super(resData, self).__init__() 
      
        self.train_dict = train_dict 
        self.batch_size = batch_size
        self.all_pos_train=all_pos 

        self.features_fill = []
        for user_id in self.train_dict:
            self.features_fill.append(user_id)
        self.set_all=set(range(num_item))
   
    def __len__(self):  
        return math.ceil(len(self.train_dict)*1.0/self.batch_size)#这里的self.data_set_count==batch_size
         

    def __getitem__(self, idx): 
        
        user_test=[]
        item_test=[]
        split_test=[]
        for i in range(self.batch_size):#这里的self.data_set_count==batch_size 
            index_my=self.batch_size*idx+i 
            if index_my == len(self.train_dict):
                break   
            user = self.features_fill[index_my]
            item_i_list = list(self.train_dict[user])
            item_j_list = list(self.set_all-self.all_pos_train[user])
            # pdb.set_trace() 
            u_i=[user]*(len(item_i_list)+len(item_j_list))
            user_test.extend(u_i)
            item_test.extend(item_i_list)
            item_test.extend(item_j_list)  
            split_test.append([(len(item_i_list)+len(item_j_list)),len(item_j_list)]) 
           
        #实际上只用到一半去计算，不需要j的。
        return torch.from_numpy(np.array(user_test)), torch.from_numpy(np.array(item_test)), split_test  