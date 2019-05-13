#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 09:40:59 2019

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


from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

from sacred import Experiment

from transformer.Models import Transformer
from transformer.Optim import ScheduledOptim
from My_models import Multi_Transformer
from tqdm import tqdm

ex = Experiment('multimodal_transformer')
from sacred.observers import MongoObserver
from global_configs import *
#The first database was multi_trans

#We must change url to the the bluehive node on which the mongo server is running
url_database = conf_url_database
mongo_database_name = conf_mongo_database_name
ex.observers.append(MongoObserver.create(url= url_database ,db_name= mongo_database_name))

@ex.config
def cfg():
    node_index = 0
    #train_batch_size = random.choice([64,128,256,512])
    train_batch_size = 256

    epoch = 400 #paul did 50
    shuffle = True
    num_workers = 2
    best_model_path =  "/scratch/echowdh2/saved_models_from_projects/multimodal_transformer/"+str(node_index) +"_best_model.chkpt"
    input_modalities_sizes = None
    dataset_location = None
    dataset_name = None
    max_seq_len=None
    
    padding_value = 0.0
    
    #To ensure that it captures the whole batch at the same time
    #and hence we get same score as Paul
    #TODO: Must cahange
    dev_batch_size=250
    test_batch_size=700

    encoder = {'d_word_vec':512,'d_model':512,'d_inner_hid':2048,'d_k':64,'d_v':64,'n_head':8,'n_layers':6,
                   'n_warmup_steps':16000,'dropout':0.1,'embs_share_weight':True,'proj_share_weight':True,
                   'label_smoothing': True,'max_token_seq_len':60,'n_source_features':325
                   }
    post_decoder = {'fc1_in':encoder["d_model"],'fc1_out':random.choice([64,128,256]), 'fc1_drop':random.choice([0.0,0.1,0.2]), 'fc2_out':1}
    post_merger = {
            "lstm_input":encoder["d_model"],
            'lstm_hidden':random.choice([32,64,128,256]),
            'fc1_output':random.choice([32,64,128,256]),
            'dropout':random.choice([0.0,0.2,0.5,0.7]),
            'fc2_output':1
            }
    loss_function="ll1"
    rand_test = [random.randrange(0,5000) for i in range(20)]
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    log = None
    save_model = "best_model"
    save_mode = 'best'
    
    prototype=False
    if prototype:
        epoch=1

class MosiDataset(Dataset):
    def __init__(self, X, Y,config):
        self.X = X
        self.Y = Y
        self.config=config
        
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        X = torch.FloatTensor(self.X[idx])
        
        #print(self.X[idx].shape)
#        numpy_seq_k = seq.numpy()
        padding_rows = np.where(~self.X[idx].any(axis=1))[0]
        n_rem_entries= self.X[idx].shape[0] - len(padding_rows)
       
        X_pos = np.concatenate(( np.zeros((len(padding_rows),)), np.array([pos+1 for pos in range(n_rem_entries)])))
        #print("X_pos:",X_pos," Len:",X_pos.shape)
        X_pos = torch.LongTensor(X_pos)       
        Y = torch.FloatTensor([self.Y[idx]])
        
        #target_X = torch.zeros(1)
        #target_X_pos = torch.zeros(1)
        #We cannot use zeros since it will output ones. But I believe the systmes will be able to learn it.
        target_X = torch.ones(1)
        target_X_pos = torch.ones(1)
        
        #So, now we will need to make it into three tensors, each corresponding to one modality. First, we add a new dimension and then expand into that dim
        #print("X",X.size(),"X_pos:",X_pos.size())
        #The size of X is (max_seq_len,num_feat) = (20,325). We insert a fake dim-1 to make it (20,1,325) and then repeat the data thrice along dim -1 to make it (20,3,325)
        new_X = X.unsqueeze(1).repeat(1,3,1)
        #As dim-1(3) denotes modality(text,audio,video) and not all data are valid, we will need to make some data zero.
        lang_d,aud_d,vid_d = self.config['input_modalities_sizes']
        
        #maxing out lang+video from audio part
        #making upto lang_d zero
        new_X[:,1,:lang_d]=0
        #Then making the vide0 entries zero
        new_X[:,1,lang_d+aud_d:]=0
        #muxing out text+zudio for video entries
        new_X[:,2,:lang_d+aud_d]=0
        #print("impact area:",new_X[:,2,-22:])
        #We need to return a seq_len,feat from here. So, we are collapsing the first two dims
        X = new_X.reshape(-1,new_X.shape[-1])
        X_pos = X_pos.unsqueeze(-1).repeat(1,3).reshape(-1)
        #print(new_X,new_X.size())
        #print(X_pos,X_pos.size())
        #assert False
        
        
        
        return X,X_pos,target_X,target_X_pos,Y 


@ex.capture        
def load_saved_data(_config):
    """
    Loads the data appropriately from the files in /data folder.
    This folder must be present in the same folder as the code for this to work
    """
    data_path = os.path.join(_config["dataset_location"],'data')
    #TODO:Change it properly
    h5f = h5py.File(os.path.join(data_path,'X_train.h5'),'r')
    X_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('data/y_train.h5','r')
    y_train = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('data/X_valid.h5','r')
    X_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('data/y_valid.h5','r')
    y_valid = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('data/X_test.h5','r')
    X_test = h5f['data'][:]
    h5f.close()
    h5f = h5py.File('data/y_test.h5','r')
    y_test = h5f['data'][:]
    h5f.close()
    return X_train, y_train, X_valid, y_valid, X_test, y_test
@ex.capture
def train_epoch(model, training_data, criterion,optimizer, device, smoothing,_config):
    ''' Epoch operation in training phase'''

    model.train()

    epoch_loss = 0.0
    num_batches = 0
   
    for batch in tqdm(training_data, mininterval=2,desc='  - (Training)   ', leave=False):

     
        X,X_pos,target_X,target_X_pos,Y  = map(lambda x: x.to(device), batch)
        #If we use "bce", we need to interpret labels as probability.
        if(_config["loss_function"] == "bce"):
            Y = nn.Sigmoid().forward(Y)
        #print("Y:",Y)
        
        
        #print("\nX:",X.size(),", X_pos:",X_pos.size()," Y:",Y.size()," T_X:",target_X.size()," T_X_pos:",target_X_pos.size())
        # forward
        optimizer.zero_grad()
        predictions = model(X,X_pos,target_X,target_X_pos,Y)
        #print(predictions.size(),train_Y.size())
        #print("pred:",predictions)

        loss = criterion(predictions, Y)
        loss.backward()
        #optimizer.step()
        epoch_loss += loss.item()

        # update parameters
        optimizer.step_and_update_lr()
        
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
    
         
            X,X_pos,target_X,target_X_pos,Y = map(lambda x: x.to(device), batch)
            #If we use "bce", we need to interpret labels as probability.
            if(_config["loss_function"] == "bce"):
                Y = nn.Sigmoid().forward(Y)
            predictions = model(X,X_pos,target_X,target_X_pos,Y)
            loss = criterion(predictions, Y)
            
            epoch_loss += loss.item()
            
            num_batches +=1
    return epoch_loss / num_batches
@ex.capture
def reload_model_from_file(file_path):
        checkpoint = torch.load(file_path)
        _config = checkpoint['_config']
        
        encoder_config = _config["encoder"]
        model = Multi_Transformer(
        
        n_src_features = encoder_config["n_source_features"],
        len_max_seq = encoder_config["max_token_seq_len"],
        _config = _config,
        tgt_emb_prj_weight_sharing=encoder_config["proj_share_weight"],
        emb_src_tgt_weight_sharing=encoder_config["embs_share_weight"],
        d_k=encoder_config["d_k"],
        d_v=encoder_config["d_v"],
        d_model=encoder_config["d_model"],
        d_word_vec=encoder_config["d_word_vec"],
        d_inner=encoder_config["d_inner_hid"],
        n_layers=encoder_config["n_layers"],
        n_head=encoder_config["n_head"],
        dropout=encoder_config["dropout"]
        ).to(_config["device"])
        

        

        model.load_state_dict(checkpoint['model'])
        print('[Info] Trained model state loaded.')

        return model
        
@ex.capture
def test_epoch(model,data_loader,criterion, device):
    ''' Epoch operation in evaluation phase '''
    epoch_loss = 0.0
    num_batches=0
    model.eval()
    returned_Y = None
    returned_predictions = None
    with torch.no_grad():
   
        for batch in tqdm(data_loader, mininterval=2,desc='  - (Validation)   ', leave=False):
    
         
            X,X_pos,target_X,target_X_pos,Y = map(lambda x: x.to(device), batch)
            predictions = model(X,X_pos,target_X,target_X_pos,Y)
            loss = criterion(predictions, Y)
            
            epoch_loss += loss.item()
            
            num_batches +=1
            #if we don'e do the squeeze, it remains as 2d numpy arraya nd hence
            #creates problems like nan while computing various statistics on them
            returned_Y = Y.squeeze(1).cpu().numpy()
            returned_predictions = predictions.squeeze(1).cpu().data.numpy()

    return returned_predictions,returned_Y   


    
            
@ex.capture
def train(model, training_data, validation_data,test_data_loader,optimizer,criterion,_config,_run):
    ''' Start training '''
    model_path = _config["best_model_path"]

    valid_losses = []
    for epoch_i in range(_config["epoch"]):
        #print('[ Epoch', epoch_i, ']')

        
        train_loss = train_epoch(
            model, training_data, criterion,optimizer, device = _config["device"], smoothing=_config["encoder"]["label_smoothing"])
        #print("\nepoch:{},train_loss:{}".format(epoch_i,train_loss))
        _run.log_scalar("training.loss", train_loss, epoch_i)


        valid_loss = eval_epoch(model, validation_data, criterion,device=_config["device"])
        _run.log_scalar("dev.loss", valid_loss, epoch_i)
        
        #scheduler.step(valid_loss)

        
        
        valid_losses.append(valid_loss)
        print("\nepoch:{},train_loss:{}, valid_loss:{}".format(epoch_i,train_loss,valid_loss))

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            '_config': _config,
            'epoch': epoch_i}

        if _config["save_model"]:
            # if _config["save_mode"] == 'all':
            #     model_name = _config["save_model"] + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_loss)
            #     torch.save(checkpoint, model_name)
            if _config["save_mode"] == 'best':
                #print(_run.experiment_info)
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_path)
                    print('    - [Info] The checkpoint file has been updated.')
            test_accuracy = test_score_model(model,test_data_loader,criterion)
            _run.log_scalar("test_per_epoch.acc", test_accuracy, epoch_i)
    #After the entire training is over, save the best model as artifact in the mongodb
    
    ex.add_artifact(model_path)

@ex.capture
def test_score_model(model,test_data_loader,criterion,_config,_run):
    
    predictions,y_test = test_epoch(model,test_data_loader,criterion,_config["device"])
    #print("predictions:",predictions,predictions.shape)
    #print("ytest:",y_test,y_test.shape)
    mae = np.mean(np.absolute(predictions-y_test))
    print("mae: ", mae)
    
    corr = np.corrcoef(predictions,y_test)[0][1]
    print("corr: ", corr)
    
    mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
    print("mult_acc: ", mult)
    
    f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
    print("mult f_score: ", f_score)
    
    #As we canged the "Y" as probability, now we need to choose yes for >=0.5
    if(_config["loss_function"]=="bce"):
        true_label = (y_test >= 0.5)
    elif(_config["loss_function"]=="ll1"):
        true_label = (y_test >= 0)
        
    predicted_label = (predictions >= 0)
    print("Confusion Matrix :")
    confusion_matrix_result = confusion_matrix(true_label, predicted_label)
    print(confusion_matrix_result)
    
    print("Classification Report :")
    classification_report_score = classification_report(true_label, predicted_label, digits=5)
    print(classification_report_score)
    
    accuracy = accuracy_score(true_label, predicted_label)
    print("Accuracy ",accuracy )
    
    _run.info['final_result']={'accuracy':accuracy,'mae':mae,'corr':corr,"mult_acc":mult,
             "mult_f_score":f_score,"Confusion Matrix":confusion_matrix_result,
             "Classification Report":classification_report_score}
    return accuracy


@ex.capture
def test_score(test_data_loader,criterion,_config,_run):
    model_path =  _config["best_model_path"]
    model = reload_model_from_file(model_path)

    predictions,y_test = test_epoch(model,test_data_loader,criterion,_config["device"])
    #print("predictions:",predictions,predictions.shape)
    #print("ytest:",y_test,y_test.shape)
    mae = np.mean(np.absolute(predictions-y_test))
    print("mae: ", mae)
    
    corr = np.corrcoef(predictions,y_test)[0][1]
    print("corr: ", corr)
    
    mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
    print("mult_acc: ", mult)
    
    f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
    print("mult f_score: ", f_score)
    
    true_label = (y_test >= 0)
    predicted_label = (predictions >= 0)
    print("Confusion Matrix :")
    confusion_matrix_result = confusion_matrix(true_label, predicted_label)
    print(confusion_matrix_result)
    
    print("Classification Report :")
    classification_report_score = classification_report(true_label, predicted_label, digits=5)
    print(classification_report_score)
    
    accuracy = accuracy_score(true_label, predicted_label)
    print("Accuracy ",accuracy )
    
    _run.info['final_result']={'accuracy':accuracy,'mae':mae,'corr':corr,"mult_acc":mult,
             "mult_f_score":f_score,"Confusion Matrix":confusion_matrix_result,
             "Classification Report":classification_report_score}
    return accuracy

@ex.capture
def set_up_data_loader(_config):
    train_X,train_Y,dev_X,dev_Y,test_X,test_Y = load_saved_data()
    
    #MUST remove it
    if(_config["prototype"]):
        train_X = train_X[:10,:,:]
        train_Y = train_Y[:10]
        
        dev_X = dev_X[:10,:,:]
        dev_Y = dev_Y[:10]
        
        test_X = test_X[:10,:,:]
        test_Y = test_Y[:10]
        
        
    train_dataset = MosiDataset(train_X,train_Y,_config)
    dev_dataset = MosiDataset(dev_X,dev_Y,_config)
    test_dataset = MosiDataset(test_X,test_Y,_config)
    
    train_dataloader = DataLoader(train_dataset, batch_size=_config["train_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    dev_dataloader = DataLoader(dev_dataset, batch_size=_config["dev_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    test_dataloader = DataLoader(test_dataset, batch_size=_config["test_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    
    #print(train_X.shape,train_Y.shape,dev_X.shape,dev_Y.shape,test_X.shape,test_Y.shape)
    #data_loader = test_data_loader(train_X,train_Y,_config)
    return train_dataloader,dev_dataloader,test_dataloader

@ex.capture
def set_random_seed(seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    Parameter:
        seed: It is set in @ex.config and will be passed through variable injection.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@ex.capture

def prep_for_training(_config):
    encoder_config = _config["encoder"]
    model = Multi_Transformer(
        
        n_src_features = encoder_config["n_source_features"],
        len_max_seq = encoder_config["max_token_seq_len"],
        _config = _config,
        tgt_emb_prj_weight_sharing=encoder_config["proj_share_weight"],
        emb_src_tgt_weight_sharing=encoder_config["embs_share_weight"],
        d_k=encoder_config["d_k"],
        d_v=encoder_config["d_v"],
        d_model=encoder_config["d_model"],
        d_word_vec=encoder_config["d_word_vec"],
        d_inner=encoder_config["d_inner_hid"],
        n_layers=encoder_config["n_layers"],
        n_head=encoder_config["n_head"],
        dropout=encoder_config["dropout"]
        ).to(_config["device"])
        

    optimizer = ScheduledOptim(
        optim.Adam(
            filter(lambda x: x.requires_grad, model.parameters()),
            betas=(0.9, 0.98), eps=1e-09),
        encoder_config["d_model"], encoder_config["n_warmup_steps"])
    
    if(_config["loss_function"]=="bce"):
            print("using bce loss")
            criterion = nn.BCEWithLogitsLoss()
    else:
        criterion = nn.L1Loss()
    criterion = criterion.to(_config["device"])
    
    # optimizer =  optim.Adam(
    #         filter(lambda x: x.requires_grad, transformer.parameters()),lr = _config["learning_rate"],
    #         betas=(0.9, 0.98), eps=1e-09)
    #torch.optim.Adam(params, lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
    #optimizer = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=False)
    #scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)
    return model,optimizer,criterion

@ex.automain
def driver(_config):
    
    set_random_seed(_config["seed"])
    #print(_config["rand_test"],_config["seed"])
    train_data_loader,dev_data_loader,test_data_loader = set_up_data_loader()
    model,optimizer,criterion = prep_for_training()
    print("Model ready")

    train(model, train_data_loader,dev_data_loader,test_data_loader ,optimizer,criterion)
    test_accuracy = test_score(test_data_loader,criterion)
    ex.log_scalar("test.accuracy",test_accuracy)
    results = dict()
    #I believe that it will try to minimize the rest. Let's see how it plays out
    results["optimization_target"] = 1 - test_accuracy

    return results




