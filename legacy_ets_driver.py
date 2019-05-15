#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 07:59:01 2019

@author: echowdh2
"""
import faulthandler
faulthandler.enable()
import sys
import numpy as np
import random
import torch
import tqdm
import os
import logging

from global_configs import *

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset


import h5py
import time
from collections import defaultdict, OrderedDict
import argparse
import pickle
import time
import json, os, ast, h5py
import math

#from models import MFN
from models import Multimodal_Video_transformer
from models import Transformed_mfn
from  Own_Optimizer import Optimizer_Scheduler 


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

from sacred import Experiment
from tqdm import tqdm
from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim

ex = Experiment('multimodal_ets')
from sacred.observers import MongoObserver

#We must change url to the the bluehive node on which the mongo server is running
url_database = conf_url_database
mongo_database_name = conf_mongo_database_name


ex.observers.append(MongoObserver.create(url= url_database ,db_name= mongo_database_name))

my_logger = logging.getLogger()
my_logger.disabled=True

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

@ex.config
def cfg():
    node_index = 0
    epoch = 10 #paul did 50
    shuffle = True
    num_workers = 2
    best_model_path =  "/scratch/echowdh2/saved_models_from_projects/ets_acii_19/"+str(node_index) +"_best_model.chkpt"
    num_context_sequence=5
    experiment_config_index=0
    
    loss_function="ll1"
    
    dataset_location = None
    dataset_name = None
    text_indices = None
    audio_indices=None
    video_indices = None
    max_num_sentences=None
    max_seq_len = None
    input_dims=None #organized as [t,a,v]
    embedded_input_dims = None
    mfn_input_dims=None
    Y_size=None
    target_label_index=None
    y_score_median_values = [5.6, 3.6, 3.8, 3.8, 4. , 3.8]
    use_positional_attention = False
    model=None
    
    inference=False
    

    
    padding_value = 0.0
    
   
    #To ensure that it captures the whole batch at the same time
    #and hence we get same score as Paul
    #TODO: Must cahange
    train_batch_size = random.choice([8])
    #These two are coming from running_different_configs.py
    dev_batch_size=8
    test_batch_size=8
    inference_batch_size = 8
    #originally, sizes of (train,dev,test) = (1200, 200, 338), but gpu has very limited space. so we need to make batch size very small
    
    

   
    
   
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    use_text=True
    use_audio=True
    use_video = True
    
   
    
    
    
    save_model = True
    save_mode = 'best'
    
    prototype=None
    
    prot_train=10
    prot_dev=10
    prot_test=10
    prot_inference = 10
    
    if prototype:
        epoch=1
       
        
    #TODO: May have to change the hidden_sizes to match with later stages
    #TODO:Will need to add RANDOM CHOICE FOR hidden_size later
    #Basically the hidden_sizes is an arry containing hidden_size for all four [t_old,a,v,t_embedded]
    #THe LSTM will get the embedded text vector,so we are using the last one 
    hidden_text =random.choice([32,64,88,128,156,256])
    hidden_audio = random.choice([8,16,32,48,64,80])
    hidden_video = random.choice([8,16,32,48,64,80])
    
    word_transformer_configs = {'d_word_vec':512,'d_model':512,'d_inner_hid':random.choice([1024,2048]),
                    'd_k':64,'d_v':64,'n_head':random.choice([1,2,3,4,5,6]),'n_layers':random.choice([1,2,3,4,5,6]),'n_warmup_steps':4000,
                    'dropout':0.1,'embs_share_weight':True,'proj_share_weight':True,
                    'label_smoothing': True,'max_token_seq_len':3,
                    'n_source_features':300}
    
    sentence_transformer_configs = {'d_word_vec':512,'d_model':512,'d_inner_hid':2048,
                    'd_k':64,'d_v':64,'n_head':random.choice([4,8]),'n_layers':random.choice([3,6]),'n_warmup_steps':random.choice([600,1200,2000,4000,8000]),
                    'dropout':0.1,'embs_share_weight':True,'proj_share_weight':True,
                    'label_smoothing': True,'max_token_seq_len':max_seq_len,
                    'n_source_features':word_transformer_configs["d_model"]*3,'h_sent_feat_lstm':random.choice([128,256,512])}
    #
    #sum(embedded_input_dims)
    
    video_transformer_configs = {'d_word_vec':512,'d_model':512,'d_inner_hid':random.choice([1024,2048]),
                    'd_k':64,'d_v':64,'n_head':random.choice([1]),'n_layers':random.choice([1]),'n_warmup_steps':4000,
                    'dropout':0.1,'embs_share_weight':True,'proj_share_weight':True,
                    'label_smoothing': True,'max_token_seq_len':max_num_sentences,
                    'n_source_features':sentence_transformer_configs['h_sent_feat_lstm'],'h_video_feat_lstm':random.choice([128,256,512])}
     
    
    #lr = random.choice(np.logspace(0.01,0.3))
    #lr=0.000000000000001
    optim = "transformer"
    
    #All these are mfn configs    
    config = dict()
    config["input_dims"] = mfn_input_dims
    hl = random.choice([32,64,88,128,156,256])
    ha = random.choice([8,16,32,48,64,80])
    hv = random.choice([8,16,32,48,64,80])
    config["h_dims"] = [hl,ha,hv]
    config["memsize"] = random.choice([64,128,256,300,400])
    config["windowsize"] = 2
    config["batchsize"] = random.choice([32,64,128,256])
    config["num_epochs"] = 50
    config["lr"] = random.choice([0.001,0.002,0.005,0.008,0.01])
    config["momentum"] = random.choice([0.1,0.3,0.5,0.6,0.8,0.9])
    
    NN1Config = dict()
    NN1Config["shapes"] = random.choice([32,64,128,256])
    NN1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    NN2Config = dict()
    NN2Config["shapes"] = random.choice([32,64,128,256])
    NN2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    gamma1Config = dict()
    gamma1Config["shapes"] = random.choice([32,64,128,256])
    gamma1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    gamma2Config = dict()
    gamma2Config["shapes"] = random.choice([32,64,128,256])
    gamma2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    outConfig = dict()
    outConfig["shapes"] = random.choice([32,64,128,256])
    outConfig["drop"] = random.choice([0.0,0.2,0.5,0.7])
    
    mfn_configs = [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig]
    
    trans_mfn_configs={
            'in_merge_sent':1,
            'h_merge_sent':random.choice([1])
            }
    post_mfn_trans_configs = {'d_word_vec':512,'d_model':512,'d_inner_hid':random.choice([1024,2048]),
                    'd_k':64,'d_v':64,'n_head':random.choice([1]),'n_layers':random.choice([1,2]),'n_warmup_steps':4000,
                    'dropout':0.1,'embs_share_weight':True,'proj_share_weight':True,
                    'label_smoothing': True,'max_token_seq_len':max_num_sentences,
                    'n_source_features':outConfig["shapes"],'fc1_out':random.choice([64,128,256,512]),"fc1_drop":random.choice([0.0,0.1,0.2,0.3])\
                    ,'fc2_out':1}
     
    
class ETSDataset(Dataset):
    
    def __init__(self,id_list,_config,all_data):
        self.id_list = id_list
        self.config=_config
        data_path = _config["dataset_location"]
            
        (self.word_aligned_facet_sdk,self.word_aligned_covarep_sdk,self.word_embedding_idx_sdk,self.y_labels) = all_data

        
        self.glove_d = 1
        self.covarep_d=81
        self.facet_d=35
        self.tot_feat_d = self.glove_d+self.covarep_d+self.facet_d

        self.max_video_len=_config["max_num_sentences"]
        self.max_sen_len=_config["max_seq_len"]
    
    def paded_word_idx(self,seq,max_sen_len=20,left_pad=1):
        seq=seq[0:max_sen_len]
        pad_w=np.concatenate((np.zeros(max_sen_len-len(seq)),seq),axis=0)
        pad_w=np.array([[w_id] for  w_id in pad_w])
        return pad_w

    def padded_covarep_features(self,seq,max_sen_len=20,left_pad=1):
        seq=seq[0:max_sen_len]
        return np.concatenate((np.zeros((max_sen_len-len(seq),self.covarep_d)),seq),axis=0)

    def padded_facet_features(self,seq,max_sen_len=20,left_pad=1):
        seq=seq[0:max_sen_len]
        
        #print("padded facet:",np.zeros(((max_sen_len-len(seq)),self.facet_d)).shape,np.array(seq).shape)
        padding = np.zeros(((max_sen_len-len(seq)),self.facet_d))
        #seq = np.array(seq)
        #print("right before concat:",padding.shape,seq.shape)
        
        ret_val =  np.concatenate((padding,seq),axis=0)
        #print("done:",ret_val.shape)
        return ret_val

    def padded_context_features(self,context_w,context_of,context_cvp,max_num_sentence,max_sen_len):
        context_w=context_w[-max_num_sentence:]
        context_of=context_of[-max_num_sentence:]
        context_cvp=context_cvp[-max_num_sentence:]

        padded_context=[]
        for i in range(len(context_w)):
            p_seq_w=self.paded_word_idx(context_w[i],max_sen_len)
            p_seq_cvp=self.padded_covarep_features(context_cvp[i],max_sen_len)
            #print("NOw processing:",np.array(context_of[i]).shape)
            p_seq_of=self.padded_facet_features(context_of[i],max_sen_len)
            #print("processed it")
            padded_context.append(np.concatenate((p_seq_w,p_seq_cvp,p_seq_of),axis=1))
            #print("and it")

        pad_c_len=max_num_sentence-len(padded_context)
        padded_context=np.array(padded_context)
        
        if not padded_context.any():
            return np.zeros((max_num_sentence,max_sen_len,self.tot_feat_d))
        #print("padded",padded_context.shape)
        return np.concatenate((np.zeros((pad_c_len,max_sen_len,self.tot_feat_d)),padded_context),axis=0)
    
        
    
    def __len__(self):
        return len(self.id_list)
    
    def __getitem__(self,index):
        
        hid=self.id_list[index]
        #print("The key is:",hid)
        text=np.array(self.word_embedding_idx_sdk[hid]['features'])
        video=np.array(self.word_aligned_facet_sdk[hid]['features'])
        audio=np.array(self.word_aligned_covarep_sdk[hid]['features'])
        
        #print("aud:",np.array(audio).shape)
        
        X=torch.FloatTensor(self.padded_context_features(text,video,audio,self.max_video_len,self.max_sen_len))
        
        X_word_pos = np.zeros((X.shape[0],X.shape[1]))
        
        for i in range(X.shape[0]):
            
            word_X = X[i,:,:]
            word_X = word_X.reshape(-1,word_X.shape[-1])
            #Then we check where we need to pad
            padding_rows = np.where(~word_X.cpu().numpy().any(axis=1))[0]
            n_rem_entries= word_X.shape[0] - len(padding_rows)
            #Then, we simple add the padding entries
            cur_X_word_pos = np.concatenate(( np.zeros((len(padding_rows),)), np.array([pos+1 for pos in range(n_rem_entries)])))
            #After that, we need to reshape
            X_word_pos[i,:] = cur_X_word_pos
        #my_logger.debug("X_pos:",X_pos," Len:",X_pos.shape)
        X_word_pos = torch.LongTensor(X_word_pos) 
        
        
        sentence_X = X.reshape(X.shape[0],-1)
        padding_rows = np.where(~sentence_X.cpu().numpy().any(axis=1))[0]
        n_rem_entries= sentence_X.shape[0] - len(padding_rows)
        #Then, we simple add the padding entries
        X_sentence_pos = np.concatenate(( np.zeros((len(padding_rows),)), np.array([pos+1 for pos in range(n_rem_entries)])))
        X_sentence_pos = torch.LongTensor(X_sentence_pos) 
        
        #an extra [] is necessary since we are getting a float this time
        Y=torch.FloatTensor([self.y_labels["labels"][hid][self.config["target_label_index"]]])
        
        if(self.config["loss_function"] !='ll1'):
            label_index = self.config["target_label_index"]

            target_median_val = self.config["y_score_median_values"][label_index]

            Y= (Y>= target_median_val)
            #We are doing it for "soft" labeling
            #Y = torch.sigmoid(Y - target_median_val)

            
                
        return X,X_word_pos,X_sentence_pos,Y

class Generic_Dataset(Dataset):
    def __init__(self, X, Y,_config):
        self.X = X
        self.Y = Y
        self.config = _config
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        
       
        
        #first, we will just reshape as (al_else,word_feature)
        #TODO:Solve it.
        X_word_pos = np.zeros((X.shape[0],X.shape[1]))
        
        for i in range(X.shape[0]):
            
            word_X = X[i,:,:]
            word_X = word_X.reshape(-1,word_X.shape[-1])
            #Then we check where we need to pad
            padding_rows = np.where(~word_X.cpu().numpy().any(axis=1))[0]
            n_rem_entries= word_X.shape[0] - len(padding_rows)
            #Then, we simple add the padding entries
            cur_X_word_pos = np.concatenate(( np.zeros((len(padding_rows),)), np.array([pos+1 for pos in range(n_rem_entries)])))
            #After that, we need to reshape
            X_word_pos[i,:] = cur_X_word_pos
        #my_logger.debug("X_pos:",X_pos," Len:",X_pos.shape)
        X_word_pos = torch.LongTensor(X_word_pos) 
        
        
        sentence_X = X.reshape(X.shape[0],-1)
        padding_rows = np.where(~sentence_X.cpu().numpy().any(axis=1))[0]
        n_rem_entries= sentence_X.shape[0] - len(padding_rows)
        #Then, we simple add the padding entries
        X_sentence_pos = np.concatenate(( np.zeros((len(padding_rows),)), np.array([pos+1 for pos in range(n_rem_entries)])))
        X_sentence_pos = torch.LongTensor(X_sentence_pos) 
        

        Y = torch.FloatTensor([self.Y[idx]])
        
        #my_logger.debug("The new Y:",Y)
        
        return X,X_word_pos,X_sentence_pos,Y




@ex.capture        
def load_saved_data(_config):
    
    data_path = os.path.join(_config["dataset_location"],'data')
    #TODO:Change it properly
    
    h5f = h5py.File(os.path.join(data_path,'X_train.h5'),'r')
    X_train = h5f['data'][:]
    h5f.close()
    
    
    h5f = h5py.File(os.path.join(data_path,'y_train.h5'),'r')
    y_train = h5f['data'][:]
    h5f.close()
    
    
    h5f = h5py.File(os.path.join(data_path,'X_valid.h5'),'r')
    X_valid = h5f['data'][:]
    h5f.close()
    
    
    h5f = h5py.File(os.path.join(data_path,'y_valid.h5'),'r')
    y_valid = h5f['data'][:]
    h5f.close()
    
    h5f = h5py.File(os.path.join(data_path,'X_test.h5'),'r')
    X_test = h5f['data'][:]
    h5f.close()
    
    
    h5f = h5py.File(os.path.join(data_path,'y_test.h5'),'r')
    y_test = h5f['data'][:]
    h5f.close()
    
    return X_train, y_train, X_valid, y_valid, X_test, y_test

@ex.capture
def temp_set_up_data_loader(_config):
    
 
    
    train_X=np.random.rand(_config["train_batch_size"],_config["max_num_sentences"],_config["max_seq_len"],sum(_config["input_dims"]))
    train_Y = np.random.rand(_config["train_batch_size"],_config['Y_size'])
    
    dev_X=np.random.rand(_config["dev_batch_size"],_config["max_num_sentences"],_config["max_seq_len"],sum(_config["input_dims"]))
    dev_Y = np.random.rand(_config["dev_batch_size"],_config['Y_size'])

    
    test_X=np.random.rand(_config["test_batch_size"],_config["max_num_sentences"],_config["max_seq_len"],sum(_config["input_dims"]))
    test_Y = np.random.rand(_config["test_batch_size"],_config['Y_size'])

    
    if(_config["prototype"]):
        train_X=train_X[:10]
        train_Y=train_Y[:10]

        
        dev_X=dev_X[:10]
        dev_Y=dev_Y[:10]

        
        test_X=test_X[:10]
        test_Y=test_Y[:10]

        
    training_set = Generic_Dataset(train_X,train_Y,_config)
    dev_set = Generic_Dataset(dev_X,dev_Y,_config)
    test_set = Generic_Dataset(test_X,test_Y,_config)
   
    train_dataloader = DataLoader(training_set, batch_size=_config["train_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    dev_dataloader = DataLoader(dev_set, batch_size=_config["dev_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    test_dataloader = DataLoader(test_set, batch_size=_config["test_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    
   
    return train_dataloader,dev_dataloader,test_dataloader

@ex.capture
def set_up_unified_dataloader(_config):
    dataset_id_file= os.path.join(_config["dataset_location"], "revised_id_list.pkl")
    dataset_id=load_pickle(dataset_id_file)
    train=dataset_id['train']
    dev=dataset_id['dev']
    test=dataset_id['test']
    all_data_id = train + dev + test
    
    if(_config["prototype"]):
        prot_num = _config["prot_inference"]
        all_data_id=all_data_id[:prot_num]
       
    
    data_path = _config["dataset_location"]    
    facet_file= os.path.join(data_path,'revised_facet.pkl')
    covarep_file=os.path.join(data_path,"covarep.pkl")
    word_vec_file=os.path.join(data_path,"glove_index.pkl")
    y_labels = os.path.join(data_path,"video_labels.pkl")
        
    word_aligned_facet_sdk=load_pickle(facet_file)
    word_aligned_covarep_sdk=load_pickle(covarep_file)
    word_embedding_idx_sdk=load_pickle(word_vec_file)
    y_labels_sdk = load_pickle(y_labels)
    all_data = (word_aligned_facet_sdk,word_aligned_covarep_sdk,word_embedding_idx_sdk,y_labels_sdk)
    
    all_dataset = ETSDataset(all_data_id,_config,all_data)
   
    all_dataloader = DataLoader(all_dataset, batch_size=_config["inference_batch_size"],
                        shuffle=False, num_workers=_config["num_workers"])
    return all_dataloader,all_data_id

@ex.capture
def set_up_data_loader(_config):
    
    dataset_id_file= os.path.join(_config["dataset_location"], "revised_id_list.pkl")
    dataset_id=load_pickle(dataset_id_file)
    train=dataset_id['train']
    dev=dataset_id['dev']
    test=dataset_id['test']
    #print("real sizes:",len(train),len(dev),len(test))
    if(_config["prototype"]):
        train_num = _config["prot_train"]
        dev_num = _config["prot_dev"]
        test_num = _config["prot_test"]
        #dev=dataset_id['train']
        


        train=train[:train_num]
        dev=dev[:dev_num]
        test=test[:test_num]
        #print("train:",train)
        #print("dev:",dev)
    
    data_path = _config["dataset_location"]    
    facet_file= os.path.join(data_path,'revised_facet.pkl')
    covarep_file=os.path.join(data_path,"covarep.pkl")
    word_vec_file=os.path.join(data_path,"glove_index.pkl")
    y_labels = os.path.join(data_path,"video_labels.pkl")
        
    word_aligned_facet_sdk=load_pickle(facet_file)
    word_aligned_covarep_sdk=load_pickle(covarep_file)
    word_embedding_idx_sdk=load_pickle(word_vec_file)
    y_labels_sdk = load_pickle(y_labels)
    all_data = (word_aligned_facet_sdk,word_aligned_covarep_sdk,word_embedding_idx_sdk,y_labels_sdk)
    
    training_set = ETSDataset(train,_config,all_data)
    dev_set = ETSDataset(dev,_config,all_data)
    test_set = ETSDataset(test,_config,all_data)
    
    #print("dataset init")
    #print("In train dataloader:",_config["train_batch_size"])
    train_dataloader = DataLoader(training_set, batch_size=_config["train_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    dev_dataloader = DataLoader(dev_set, batch_size=_config["dev_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    test_dataloader = DataLoader(test_set, batch_size=_config["test_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    #print("data loader prepared")
    #my_logger.debug(train_X.shape,train_Y.shape,dev_X.shape,dev_Y.shape,test_X.shape,test_Y.shape)
    #data_loader = test_data_loader(train_X,train_Y,_config)
    return train_dataloader,dev_dataloader,test_dataloader

@ex.capture
def set_random_seed(_seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    Parameter:
        seed: It is set in @ex.config and will be passed through variable injection.
    """
    np.random.seed(_seed)
    torch.manual_seed(_seed)
    torch.cuda.manual_seed(_seed)



@ex.capture
def train_epoch(model, training_data, criterion,optimizer, device,_config):
    ''' Epoch operation in training phase'''

    model.train()

    epoch_loss = 0.0
    num_batches = 0
    #sDismissed:tep_and_update_lr() is being called on every epoch. May be, it was fine when the batch size was very big. Now,the batch size is 16.
    #So, we may take one update every 30 iterations since it seems that 16 will be highest batch size and 16*30=480 which is a regular batch size
    #Or since there are 1200 data and 75 batches each with 16 data, we can take an update after every 25 batch.
   
   
    for batch in tqdm(training_data, mininterval=2,desc='  - (Training)   ', leave=False):

     #TODO: For simplicity, we are not using X_pos right now as we really do not know
     #how it can be used properly. So, we will just use the context information only.
        #X_Punchline,X_Context,X_pos_Context,Y = map(lambda x: x.to(device), batch)
        X,X_word_pos,X_sentence_pos,Y = map(lambda x: x.to(device), batch)
        #print("Train:",X)
        #print(X.size(),X_word_pos.size(),X_sentence_pos.size(),Y.size())
      
        optimizer.zero_grad()
        predictions,sent_att_mat,*_ = model(X,X_word_pos,X_sentence_pos,Y)
        
        #predictions = predictions.squeeze(0)
        #print(predictions.size(),Y.size())
        #print(sent_att_mat)
        #print(predictions,Y)
        
        if _config["loss_function"]=="bce":
            label_index = _config["target_label_index"]
    
            target_median_val = _config["y_score_median_values"][label_index]
            predictions = predictions - target_median_val
        loss = criterion(predictions,Y.float())
        loss.backward()
        #optimizer.step()
        epoch_loss += loss.item()
        
        if(_config["optim"]=="transformer"):
            optimizer.step_and_update_lr()
        elif(_config["optim"]=="paul"):
            optimizer.step_optimizer()


        num_batches +=1

   
    return epoch_loss / num_batches

@ex.capture
def eval_epoch(model,data_loader,criterion, device,_config):
    ''' Epoch operation in evaluation phase '''
    epoch_loss = 0.0
    num_batches=0
    model.eval()
    with torch.no_grad():
   
        for batch in tqdm(data_loader, mininterval=2,desc='  - (Validation)   ', leave=False):
            
            X,X_word_pos,X_sentence_pos,Y = map(lambda x: x.to(device), batch)
            #print("Dev:",X)

            predictions,*_ = model(X,X_word_pos,X_sentence_pos,Y)
            
            if _config["loss_function"]=="bce":
                label_index = _config["target_label_index"]
    
                target_median_val = _config["y_score_median_values"][label_index]
                predictions = predictions - target_median_val
            loss = criterion(predictions, Y.float())
            
            epoch_loss += loss.item()
            
            num_batches +=1
    return epoch_loss / num_batches
@ex.capture
def reload_model_from_file(file_path):
        checkpoint = torch.load(file_path)
        _config = checkpoint['_config']
        
        #encoder_config = _config["multimodal_context_configs"]
        if _config["model"]=="trans_mfn":
            model = Transformed_mfn(_config).to(_config["device"])
        else:
            model = Multimodal_Video_transformer(_config,my_logger).to(_config["device"])
        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        return model
        
@ex.capture
def test_epoch(model,data_loader,criterion, device,_config):
    ''' Epoch operation in evaluation phase '''
    epoch_loss = 0.0
    num_batches=0
    model.eval()
    returned_Y = None
    returned_predictions = None
    with torch.no_grad():
   
        for batch in tqdm(data_loader, mininterval=2,desc='  - (Validation)   ', leave=False):
    
         
            X,X_word_pos,X_sentence_pos,Y = map(lambda x: x.to(device), batch)
            
           
            predictions,*_ = model(X,X_word_pos,X_sentence_pos,Y)
            predictions = predictions.squeeze(0)  
              
            loss = criterion(predictions, Y.float())
            
            epoch_loss += loss.item()
            
            num_batches +=1
            #if we don'e do the squeeze, it remains as 2d numpy arraya nd hence
            #creates problems like nan while computing various statistics on them
            temp_Y = Y.squeeze(1).cpu().numpy()
            returned_Y = temp_Y if (returned_Y is None) else np.concatenate((returned_Y,temp_Y))

            temp_pred = predictions.squeeze(1).cpu().data.numpy()
            returned_predictions = temp_pred if returned_predictions is None else np.concatenate((returned_predictions,temp_pred))
            
    return returned_predictions,returned_Y   


    
            
@ex.capture
def train(model, training_data, validation_data, optimizer,criterion,_config,_run):
    ''' Start training '''
    model_path = _config["best_model_path"]

    valid_losses = []
    for epoch_i in range(_config["epoch"]):
        
        train_loss = train_epoch(
            model, training_data, criterion,optimizer, device = _config["device"])
        _run.log_scalar("training.loss", train_loss, epoch_i)
        


        valid_loss = eval_epoch(model, validation_data, criterion,device=_config["device"])
        _run.log_scalar("dev.loss", valid_loss, epoch_i)
        if _config["optim"]=="paul":
           optimizer.step_scheduler(valid_loss)

        
        
        valid_losses.append(valid_loss)
        print("\nepoch:{},train_loss:{}, valid_loss:{}".format(epoch_i,train_loss,valid_loss))
      #Due to space3 constraint, we are not saving the models. There should be enough info
      #in sacred to reproduce the results on the fly
        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            '_config': _config,
            'epoch': epoch_i}

        if _config["save_model"]:
            if _config["save_mode"] == 'best':
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_path)
                    print('    - [Info] The checkpoint file has been updated.')
    #After the entire training is over, save the best model as artifact in the mongodb, only if it is not protptype
    #Due to space constraint, we are not saving the model since it is not necessary as we know the seed. If we need to regenrate the result
    #simple running it again should work
    # if(_config["prototype"]==False):
    #     ex.add_artifact(model_path)


@ex.capture
def test_score_from_file(test_data_loader,criterion,_config,_run):
    model_path =  _config["best_model_path"]
    model = reload_model_from_file(model_path)

    predictions,y_test = test_epoch(model,test_data_loader,criterion,_config["device"])
    #print("test shape:",predictions.shape,y_test.shape)
    print("predictions:",predictions,predictions.shape)
    print("ytest:",y_test,y_test.shape)
    mae = np.mean(np.absolute(predictions-y_test))
    print("mae: ", mae)
    
    corr = np.corrcoef(predictions,y_test)[0][1]
    print("corr: ", corr)
    
    mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
    print("mult_acc: ", mult)
    
    if _config["loss_function"]=="ll1":
        #may be wrong
        label_index = _config["target_label_index"]
        target_median_val = _config["y_score_median_values"][label_index]
        predicted_label = (predictions >= target_median_val)
        true_label = (y_test >= target_median_val)
    else:
        label_index = _config["target_label_index"]
        target_median_val = _config["y_score_median_values"][label_index]
        predicted_label = (predictions >= target_median_val) 
        true_label = y_test
        #soft labelling may not work
        #true_label = y_test>=0.5#since they had soft labelling

    f_score = round(f1_score(np.round(predicted_label),np.round(true_label),average='weighted'),5)
    ex.log_scalar("test.f_score",f_score)

    #print("Confusion Matrix :")
    confusion_matrix_result = confusion_matrix(true_label, predicted_label)
    #print(confusion_matrix_result)
    
    #print("Classification Report :")
    classification_report_score = classification_report(true_label, predicted_label, digits=5)
    #print(classification_report_score)
    
    accuracy = accuracy_score(true_label, predicted_label)
    print("Accuracy:",accuracy )
    
    _run.info['final_result']={'accuracy':accuracy,'mae':mae,'corr':corr,"mult_acc":mult,
             "mult_f_score":f_score,"Confusion Matrix":confusion_matrix_result,
             "Classification Report":classification_report_score}
    return accuracy

@ex.capture
def inference_epoch(model,data_loader,criterion,_config):
    ''' Epoch operation in inference phase '''
   
    device = _config["device"]
    model.eval()
    all_att_weights = None
    all_pos_weights = []
    all_Y = None
    all_sentence_pos = None
    with torch.no_grad():
   
        for batch in tqdm(data_loader, mininterval=2,desc='  - (Inference)   ', leave=False):
            
            X,X_word_pos,X_sentence_pos,Y = map(lambda x: x.to(device), batch)

            pred,att_weights_mat,pos_sent_scores = model(X,X_word_pos,X_sentence_pos,Y)
            pred,att_weights_mat,pos_sent_scores = map(lambda x: x.cpu() if x is not None else None,[pred,att_weights_mat,pos_sent_scores])
            
            #print(type(att_weights_mat),type(pos_sent_scores))
            #print(att_weights_mat,att_weights_mat.size())
            #all_att_weights.append(att_weights_mat)
            if _config["use_positional_attention"]==True:
                all_pos_weights.append(pos_sent_scores)
            all_att_weights = att_weights_mat if all_att_weights is None else torch.cat((all_att_weights, att_weights_mat))
            all_Y = Y if all_Y is None else torch.cat((all_Y, Y))

            all_sentence_pos = X_sentence_pos if all_sentence_pos is None else torch.cat((all_sentence_pos,X_sentence_pos))
            
           
    return all_att_weights,all_pos_weights,all_Y,all_sentence_pos

@ex.capture
def prepare_for_training(_config):
    train_data_loader,dev_data_loader,test_data_loader = set_up_data_loader()
    
    if _config["model"]=="trans_mfn":
        model = Transformed_mfn(_config).to(_config["device"])
    else:
        model = Multimodal_Video_transformer(_config,my_logger).to(_config["device"])

    #for now, we will use the same scheduler for the entire model.
    #Later, if necessary, we may use the default optimizer of MFN
    #TODO: May have to use separate scheduler for transformer and mfn
    #We are using the optimizer and scgheduler of mfn as a last resort
    if(_config["optim"]=="transformer"):
        optimizer = ScheduledOptim(
            optim.Adam(
                filter(lambda x: x.requires_grad, model.parameters()),
                betas=(0.9, 0.98), eps=1e-09),
            _config["sentence_transformer_configs"]["d_model"]*5,_config["sentence_transformer_configs"]["n_warmup_steps"])
    elif(_config["optim"]=="paul"):
        print("initializing paul trans")
        optimizer_adam =  optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),lr = _config["lr"],
            betas=(0.9, 0.98), eps=1e-09)
    
        scheduler = ReduceLROnPlateau(optimizer_adam,mode='min',patience=100,factor=0.5,verbose=True)
        optimizer = Optimizer_Scheduler(optimizer_adam,scheduler)
        
    #We are multiplying by 3 as there are three different transformer units
    
    
    #TODO: May have to change the criterion
    #since the scores are in float format, we are using the L1Loss
    if(_config["loss_function"]=="ll1"):
        criterion = nn.L1Loss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    criterion = criterion.to(_config["device"])
    
    return train_data_loader,dev_data_loader,test_data_loader,model,optimizer,criterion

@ex.command
def run_on_a_seed_then_run_inference(_config,_run):
    print("Running inference")
    print("device:",_config["device"]," loss:",_config["loss_function"])
    set_random_seed(_config["seed"])
    train_data_loader,dev_data_loader,test_data_loader,model,optimizer,criterion = prepare_for_training()
    train(model, train_data_loader,dev_data_loader, optimizer, criterion)
    test_accuracy = test_score_from_file(test_data_loader,criterion)
    ex.log_scalar("test.accuracy",test_accuracy)
    print("training completed, now inference will begin")
    unified_data_loader,all_data_id = set_up_unified_dataloader()
    
    model_path =  _config["best_model_path"]
    model = reload_model_from_file(model_path)

    att_mat,pos_att,Y,all_sentence_pos = inference_epoch(model,unified_data_loader,criterion,_config)
    
    #print(len(att_mat),len(pos_att))
    #print(att_mat)
    res = {"attention_mat":att_mat.cpu().numpy(),"data_id":all_data_id,"Y":Y.cpu().numpy(),"sentence_pos":all_sentence_pos.cpu().numpy()}
    file_name = "inference_files/inference_" + str(_config["node_index"]) +"_"+ str(_config["seed"]) + ".pkl"
    
    out_f = open(file_name,'wb')
    pickle.dump(res,out_f)
    out_f.close()
    ex.add_artifact(file_name)
    
    


    
@ex.automain
def driver(_config,_run):
    
    set_random_seed(_config["seed"])
    train_data_loader,dev_data_loader,test_data_loader,model,optimizer,criterion = prepare_for_training()
    train(model, train_data_loader,dev_data_loader, optimizer, criterion)
    #assert False



    #test_accuracy =  test_score_from_model(model,test_data_loader,criterion)
    
    test_accuracy = test_score_from_file(test_data_loader,criterion)
    ex.log_scalar("test.accuracy",test_accuracy)
    results = dict()
    #I believe that it will try to minimize the rest. Let's see how it plays out
    results["optimization_target"] = 1 - test_accuracy

    return results

