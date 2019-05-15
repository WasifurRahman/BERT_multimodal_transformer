#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  8 16:30:43 2019

@author: echowdh2
"""


from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import pickle
import sys
sys.path.insert(0,'./pytorch-pretrained-BERT')
# from mosi_dataset_constants import SDK_PATH, DATA_PATH, WORD_EMB_PATH, CACHE_PATH
# import sys

# if SDK_PATH is None:
#     print("SDK path is not specified! Please specify first in constants/paths.py")
#     exit(0)
# else:
#     sys.path.append(SDK_PATH)
    
import numpy as np

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import (DataLoader, RandomSampler, SequentialSampler,
                              TensorDataset)
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig
from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

logger = logging.getLogger(__name__)
from sacred import Experiment

bert_ex = Experiment('bert_multimodal_transformer')
from sacred.observers import MongoObserver
from global_configs import *
url_database = conf_url_database
mongo_database_name = conf_mongo_database_name
bert_ex.observers.append(MongoObserver.create(url= url_database ,db_name= mongo_database_name))

class InputExample(object):
    """A single training/test example for simple sequence classification."""

    def __init__(self, guid, text_a, text_b=None, label=None):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            text_a: string. The untokenized text of the first sequence. For single
            sequence tasks, only this sequence must be specified.
            text_b: (Optional) string. The untokenized text of the second sequence.
            Only must be specified for sequence pair tasks.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.text_a = text_a
        self.text_b = text_b
        self.label = label
    def __str__(self):
        print("guid:{0},text_a:{1},text_b:{2},label:{3}".format(self.guid,self.text_a,self.text_b,self.label))



class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



@bert_ex.config
def cnf():
    dataset_location=None
    bert_model=None
    data_dir=None
    node_index=None
    prototype=None
    dataset_location=None
    dataset_name=None
    task_name=None
    do_train=True
    do_eval=True
    do_lower_case=True
    cache_dir=None
    max_seq_length=128
    train_batch_size=32
    learning_rate=5e-5
    num_train_epochs=20
    seed=None
    output_dir = None
    server_ip = None
    server_port=None
    eval_batch_size=8
    warmup_proportion = 0.1
    no_cuda=False
    local_rank=-1
    gradient_accumulation_steps=1
    fp16=False
    loss_scale=0
    input_modalities_sizes=None
    device = torch.device("cuda" if torch.cuda.is_available() and not no_cuda else "cpu")
    n_gpu = torch.cuda.device_count()
    output_mode=None
    label_list=None
    num_labels=len(label_list)
    dev_batch_size=None
    test_batch_size=None
    shuffle=True
    num_workers=2
    best_model_path =  "/scratch/echowdh2/saved_models_from_projects/bert_transformer/"+str(node_index) +"_best_model.chkpt"
    loss_function="ll1"
    save_model=True
    save_mode='best'
    
    d_acoustic_in=0
    d_visual_in = 0
    h_audio_lstm = 0
    h_video_lstm = 0
    h_merge_sent = 0
    

def multi_collate(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)
    
    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.cat([torch.from_numpy(sample[1]) for sample in batch], dim=0)
    sentences = pad_sequence([torch.LongTensor(sample[0][0]) for sample in batch], padding_value=PAD)
    visual = pad_sequence([torch.FloatTensor(sample[0][1]) for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2]) for sample in batch])
    
    # lengths are useful later in using RNNs
    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
    return sentences, visual, acoustic, labels, lengths


@bert_ex.capture
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,_config):
    """Loads a data file into a list of `InputBatch`s."""
    #print("label_list:",label_list)

    label_map = {label : i for i, label in enumerate(label_list)}
    with open(os.path.join(_config["dataset_location"],'word2id.pickle'), 'rb') as handle:
        word_2_id = pickle.load(handle)
    id_2_word = { id_:word for (word,id_) in word_2_id.items()}
    #print(id_2_word)
    

    features = []
    for (ex_index, example) in enumerate(examples):
       
        (words, visual, acoustic), label, segment = example
        #print(words,label, segment)
        #we will look at acoustic and visual later
        words = " ".join([id_2_word[w] for w in words])
        #print("string word:", words)
        example = InputExample(guid = segment, text_a = words, text_b=None, label=label.item())
        #print(example)
        
        tokens_a = tokenizer.tokenize(example.text_a)

        tokens_b = None
        if example.text_b:
            tokens_b = tokenizer.tokenize(example.text_b)
            # Modifies `tokens_a` and `tokens_b` in place so that the total
            # length is less than the specified length.
            # Account for [CLS], [SEP], [SEP] with "- 3"
            _truncate_seq_pair(tokens_a, tokens_b, max_seq_length - 3)
        else:
            # Account for [CLS] and [SEP] with "- 2"
            if len(tokens_a) > max_seq_length - 2:
                tokens_a = tokens_a[:(max_seq_length - 2)]

        # The convention in BERT is:
        # (a) For sequence pairs:
        #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
        #  type_ids: 0   0  0    0    0     0       0 0    1  1  1  1   1 1
        # (b) For single sequences:
        #  tokens:   [CLS] the dog is hairy . [SEP]
        #  type_ids: 0   0   0   0  0     0 0
        #
        # Where "type_ids" are used to indicate whether this is the first
        # sequence or the second sequence. The embedding vectors for `type=0` and
        # `type=1` were learned during pre-training and are added to the wordpiece
        # embedding vector (and position vector). This is not *strictly* necessary
        # since the [SEP] token unambiguously separates the sequences, but it makes
        # it easier for the model to learn the concept of sequences.
        #
        # For classification tasks, the first vector (corresponding to [CLS]) is
        # used as as the "sentence vector". Note that this only makes sense because
        # the entire model is fine-tuned.
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

       

        features.append(
                InputFeatures(input_ids=input_ids,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id))
    return features


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length."""

    # This is a simple heuristic which will always truncate the longer sequence
    # one token at a time. This makes more sense than truncating an equal percent
    # of tokens from each, since if one sequence is very short then each token
    # that's truncated likely contains more information than a longer sequence.
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_length:
            break
        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

@bert_ex.capture
def get_appropriate_dataset(data,tokenizer, output_mode,_config):
    features = convert_examples_to_features(
            data, _config["label_list"],_config["max_seq_length"], tokenizer, output_mode)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    
    #print("bert_ids:",all_input_ids)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask, all_segment_ids, all_label_ids)
    return dataset
   
        
@bert_ex.capture
def set_up_data_loader(_config):
    
    # #MUST remove it
   
    # train_examples = None
    # num_train_optimization_steps = None
    # if args.do_train:
    #     train_examples = processor.get_train_examples(args.data_dir)
    #     #print("Train examples:",train_examples)
    #     #assert False
    #     num_train_optimization_steps = int(
    #         len(train_examples) / args.train_batch_size / args.gradient_accumulation_steps) * args.num_train_epochs
    #     if args.local_rank != -1:
    #         num_train_optimization_steps = num_train_optimization_steps // torch.distributed.get_world_size()
    
        
    with open(os.path.join(_config["dataset_location"],'all_mod_data.pickle'), 'rb') as handle:
        all_data = pickle.load(handle)
    train_data = all_data["train"]
    dev_data=all_data["dev"]
    test_data=all_data["test"]
    
    if(_config["prototype"]):
        train_data=train_data[:100]
        dev_data=dev_data[:100]
        test_data=test_data[:100]     
    
    
    tokenizer = BertTokenizer.from_pretrained(_config["bert_model"], do_lower_case=_config["do_lower_case"])
    output_mode = _config["output_mode"]
    
    train_dataset = get_appropriate_dataset(train_data,tokenizer, output_mode,_config)
    dev_dataset = get_appropriate_dataset(dev_data,tokenizer, output_mode,_config)
    test_dataset = get_appropriate_dataset(test_data,tokenizer, output_mode,_config)
    
    #print("train_dataset:",train_dataset)
    #print(len(train_dataset),_config["train_batch_size"],_config["gradient_accumulation_steps"], _config["num_train_epochs"])
    num_train_optimization_steps = int(len(train_dataset) / _config["train_batch_size"] / _config["gradient_accumulation_steps"]) * _config["num_train_epochs"]
    #print("num_tr_opt_st:",num_train_optimization_steps)
    
    #print("Train len:",len(train_dataset)," dev:",len(dev_dataset)," test:",len(test_dataset))
  
    train_dataloader = DataLoader(train_dataset, batch_size=_config["train_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    dev_dataloader = DataLoader(dev_dataset, batch_size=_config["dev_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    test_dataloader = DataLoader(test_dataset, batch_size=_config["test_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    
    #print(train_X.shape,train_Y.shape,dev_X.shape,dev_Y.shape,test_X.shape,test_Y.shape)
    #data_loader = test_data_loader(train_X,train_Y,_config)
    return train_dataloader,dev_dataloader,test_dataloader,num_train_optimization_steps





@bert_ex.capture
def set_random_seed(seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    Parameter:
        seed: It is set in @ex.config and will be passed through variable injection.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@bert_ex.capture
def prep_for_training(num_train_optimization_steps,_config):
    tokenizer = BertTokenizer.from_pretrained(_config["bert_model"], do_lower_case=_config["do_lower_case"])


    # TODO:Change model here
    model = BertForSequenceClassification.from_pretrained(_config["bert_model"],
              cache_dir=_config["cache_dir"],
              num_labels=_config["num_labels"])
   
    model.to(_config["device"])
   

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    
    optimizer = BertAdam(optimizer_grouped_parameters,
                         lr=_config["learning_rate"],
                         warmup=_config["warmup_proportion"],
                         t_total=num_train_optimization_steps)
    
    return model,optimizer,tokenizer

@bert_ex.capture
def train_epoch(model,train_dataloader,optimizer,_config):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(_config["device"]) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, segment_ids, input_mask, labels=None)

            if _config["output_mode"] == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, _config["num_labels"]), label_ids.view(-1))
            elif _config["output_mode"] == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))

            
            if _config["gradient_accumulation_steps"] > 1:
                loss = loss / _config["gradient_accumulation_steps"]

            
            loss.backward()

            tr_loss += loss.item()
            nb_tr_examples += input_ids.size(0)
            nb_tr_steps += 1
            if (step + 1) % _config["gradient_accumulation_steps"] == 0:
                optimizer.step()
                optimizer.zero_grad()
                #global_step += 1   
        return tr_loss

@bert_ex.capture
def eval_epoch(model,dev_dataloader,optimizer,_config):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(_config["device"]) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, segment_ids, input_mask, labels=None)

            if _config["output_mode"] == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, _config["num_labels"]), label_ids.view(-1))
            elif _config["output_mode"] == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))

            
            if _config["gradient_accumulation_steps"] > 1:
                loss = loss / _config["gradient_accumulation_steps"]

            dev_loss += loss.item()
            nb_dev_examples += input_ids.size(0)
            nb_dev_steps += 1
            
 
    return dev_loss
   
@bert_ex.capture
def test_epoch(model,data_loader,_config):
    ''' Epoch operation in evaluation phase '''
   
            
    # epoch_loss = 0.0
    # num_batches=0
    model.eval()
    # returned_Y = None
    # returned_predictions = None
    eval_loss=0.0
    nb_eval_steps=0
    preds=[]
    all_labels=[]
    with torch.no_grad():
   
        for batch in tqdm(data_loader, mininterval=2,desc='  - (Validation)   ', leave=False):
            batch = tuple(t.to(_config["device"]) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, segment_ids, input_mask, labels=None)
            
            # create eval loss and other metric required by the task
            if _config["output_mode"] == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(logits.view(-1, num_labels), label_ids.view(-1))
            elif _config["output_mode"] == "regression":
                loss_fct = MSELoss()
                tmp_eval_loss = loss_fct(logits.view(-1), label_ids.view(-1))
            
            eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1
            if len(preds) == 0:
                preds.append(logits.detach().cpu().numpy())
                all_labels.append(label_ids.detach().cpu().numpy())
            else:
                preds[0] = np.append(
                    preds[0], logits.detach().cpu().numpy(), axis=0)
                all_labels[0] = np.append(
                    all_labels[0], label_ids.detach().cpu().numpy(), axis=0)
        eval_loss = eval_loss / nb_eval_steps
        preds = preds[0]
        all_labels=all_labels[0]
        if _config["output_mode"] == "classification":
            preds = np.argmax(preds, axis=1)
        elif _config["output_mode"] == "regression":
            preds = np.squeeze(preds)
            all_labels=np.squeeze(all_labels)
         
          
            # loss = criterion(predictions, Y)
            
            # epoch_loss += loss.item()
            
            # num_batches +=1
            # #if we don'e do the squeeze, it remains as 2d numpy arraya nd hence
            # #creates problems like nan while computing various statistics on them
            # returned_Y = Y.squeeze(1).cpu().numpy()
            # returned_predictions = predictions.squeeze(1).cpu().data.numpy()

    return preds,all_labels  
   
@bert_ex.capture
def test_score_model(model,test_data_loader,_config,_run):
    
    predictions,y_test = test_epoch(model,test_data_loader)
    #print("predictions:",predictions,predictions.shape)
    #print("ytest:",y_test,y_test.shape)
    
    mae = np.mean(np.absolute(predictions-y_test))
    #print("mae: ", mae)
    
    corr = np.corrcoef(predictions,y_test)[0][1]
    #print("corr: ", corr)
    
    mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
    #print("mult_acc: ", mult)
    
    f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
    #print("mult f_score: ", f_score)
    
    #As we canged the "Y" as probability, now we need to choose yes for >=0.5
    if(_config["loss_function"]=="bce"):
        true_label = (y_test >= 0.5)
    elif(_config["loss_function"]=="ll1"):
        true_label = (y_test >= 0)
        
    predicted_label = (predictions >= 0)
    #print("Confusion Matrix :")
    confusion_matrix_result = confusion_matrix(true_label, predicted_label)
    #print(confusion_matrix_result)
    
    #print("Classification Report :")
    classification_report_score = classification_report(true_label, predicted_label, digits=5)
    #print(classification_report_score)
    
    accuracy = accuracy_score(true_label, predicted_label)
    print("Accuracy ",accuracy )
    
    _run.info['final_result']={'accuracy':accuracy,'mae':mae,'corr':corr,"mult_acc":mult,
             "mult_f_score":f_score,"Confusion Matrix":confusion_matrix_result,
             "Classification Report":classification_report_score}
    return accuracy
            
@bert_ex.capture
def train(model, train_dataloader, validation_dataloader,test_data_loader,optimizer,_config,_run):
    ''' Start training '''
    model_path = _config["best_model_path"]

    valid_losses = []
    for epoch_i in range(int(_config["num_train_epochs"])):
           

        #print('[ Epoch', epoch_i, ']')

        
        train_loss = train_epoch(model,train_dataloader,optimizer)
        #print("\nepoch:{},train_loss:{}".format(epoch_i,train_loss))
        _run.log_scalar("training.loss", train_loss, epoch_i)


        valid_loss = eval_epoch(model, validation_dataloader,optimizer)
        _run.log_scalar("dev.loss", valid_loss, epoch_i)
        

        
        
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
                    test_accuracy = test_score_model(model,test_data_loader)
                    _run.log_scalar("test_per_epoch.acc", test_accuracy, epoch_i)
    #After the entire training is over, save the best model as artifact in the mongodb
    
    
@bert_ex.automain
def main(_config):
    
    set_random_seed(_config["seed"])
    #print(_config["rand_test"],_config["seed"])
    train_data_loader,dev_data_loader,test_data_loader,num_train_optimization_steps = set_up_data_loader()
    
    model,optimizer,tokenizer = prep_for_training(num_train_optimization_steps)
    

    train(model, train_data_loader,dev_data_loader,test_data_loader,optimizer)
    # assert False

    #TODO:need to fix it
    # test_accuracy = test_score(test_data_loader,criterion)
    # ex.log_scalar("test.accuracy",test_accuracy)
    # results = dict()
    # #I believe that it will try to minimize the rest. Let's see how it plays out
    # results["optimization_target"] = 1 - test_accuracy

    #return results
