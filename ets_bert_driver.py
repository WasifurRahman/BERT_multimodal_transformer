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
                              TensorDataset,Dataset)
#from torch.utils.data import DataLoader, Dataset

from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef

from pytorch_pretrained_bert.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_pretrained_bert.modeling import BertForSequenceClassification, BertConfig,MultimodalBertForSequenceClassification,ETSBertForSequenceClassification
#from pytorch_pretrained_bert.tokenization import BertTokenizer
#We are using the tokenization that amir did
from pytorch_pretrained_bert.amir_tokenization import BertTokenizer

from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

logger = logging.getLogger(__name__)
from sacred import Experiment

ets_bert_ex = Experiment('bert_etsr')
from sacred.observers import MongoObserver
from global_configs import *
url_database = conf_url_database
mongo_database_name = conf_mongo_database_name
ets_bert_ex.observers.append(MongoObserver.create(url= url_database ,db_name= mongo_database_name))

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
        return "guid:{0},text_a:{1},text_b:{2},label:{3}".format(self.guid,self.text_a,self.text_b,self.label)


class ETSDataset(Dataset):
    
    def __init__(self,id_list,_config,all_data,tokenizer):
        self.id_list = id_list
        self.config=_config
        self.tokenizer = tokenizer
        data_path = _config["dataset_location"]
            
        (self.word_aligned_facet_sdk,self.word_aligned_covarep_sdk,self.word_embedding_idx_sdk,self.y_labels,self.id_2_word) = all_data

        
        self.glove_d = 1
        self.covarep_d=81
        self.facet_d=35
        self.tot_feat_d = self.glove_d+self.covarep_d+self.facet_d

        self.max_video_len=_config["max_num_sentences"]
        self.max_sen_len=_config["max_seq_length"]
    
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
    
    def process_a_video(self):
        print("ok")
        
    def __getitem__(self,index):
        
            hid=self.id_list[index]
            #print("The key is:",hid)
            text=np.array(self.word_embedding_idx_sdk[hid]['features'])
            visual=np.array(self.word_aligned_facet_sdk[hid]['features'])
            acoustic=np.array(self.word_aligned_covarep_sdk[hid]['features'])
            #print("checking 0 index:{0} and text len{1}:".format(self.id_2_word[0],text.shape))
            #max_num_sentence
            #if(text.shape[0] <)
            label=torch.FloatTensor([self.y_labels["labels"][hid][self.config["target_label_index"]] - self.config["label_median"]])
            data = (text,visual,acoustic,label,hid,self.id_2_word)
            features,video_len = convert_examples_to_features(data, self.config["label_list"],self.config["max_seq_length"], self.tokenizer, self.config["output_mode"])
            #print(features)
            
            #(words, visual, acoustic), label, segment
            
            all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
            all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
            all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
            all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
            all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
    
    #print("bert_ids:",all_input_ids)

            if self.config["output_mode"] == "classification":
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
            elif self.config["output_mode"] == "regression":
                all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)
        
            # dataset = TensorDataset(all_input_ids, all_visual,all_acoustic,all_input_mask, all_segment_ids, all_label_ids)
            #print("all_input_ids:{0}, all_visual:{1},all_acoustic:{2},all_input_mask:{3}, all_segment_ids:{4}, all_label_ids:{5},video_len:{6}".format(all_input_ids.shape, all_visual.shape,all_acoustic.shape,all_input_mask.shape, all_segment_ids.shape, all_label_ids.shape,np.array([video_len]).shape))
            n_padding_rows = [self.config["max_num_sentences"] - all_input_ids.size()[0]]
            
            all_input_ids = torch.cat((all_input_ids, torch.zeros(n_padding_rows + list(all_input_ids.size()[1:]),dtype=all_input_ids.dtype)))
            all_visual = torch.cat((all_visual, torch.zeros(n_padding_rows + list(all_visual.size()[1:]),dtype=all_visual.dtype)))
            all_acoustic = torch.cat((all_acoustic, torch.zeros(n_padding_rows + list(all_acoustic.size()[1:]),dtype=all_acoustic.dtype)))
            all_input_mask = torch.cat((all_input_mask, torch.zeros(n_padding_rows + list(all_input_mask.size()[1:]),dtype=all_input_mask.dtype)))
            all_segment_ids = torch.cat((all_segment_ids, torch.zeros(n_padding_rows + list(all_segment_ids.size()[1:]),dtype=all_segment_ids.dtype)))
            #not sending it
            all_label_ids = torch.cat((all_label_ids, torch.zeros(n_padding_rows + list(all_label_ids.size()[1:]),dtype=all_label_ids.dtype)))

            
            #print(all_input_ids.size())
            #We are not sending all_label_ids
            return all_input_ids, all_visual,all_acoustic,all_input_mask, all_segment_ids, label,torch.tensor([video_len])
        
        # #print("aud:",np.array(audio).shape)
        
        # X=torch.FloatTensor(self.padded_context_features(text,video,audio,self.max_video_len,self.max_sen_len))
        
        # X_word_pos = np.zeros((X.shape[0],X.shape[1]))
        
        # for i in range(X.shape[0]):
            
        #     word_X = X[i,:,:]
        #     word_X = word_X.reshape(-1,word_X.shape[-1])
        #     #Then we check where we need to pad
        #     padding_rows = np.where(~word_X.cpu().numpy().any(axis=1))[0]
        #     n_rem_entries= word_X.shape[0] - len(padding_rows)
        #     #Then, we simple add the padding entries
        #     cur_X_word_pos = np.concatenate(( np.zeros((len(padding_rows),)), np.array([pos+1 for pos in range(n_rem_entries)])))
        #     #After that, we need to reshape
        #     X_word_pos[i,:] = cur_X_word_pos
        # #my_logger.debug("X_pos:",X_pos," Len:",X_pos.shape)
        # X_word_pos = torch.LongTensor(X_word_pos) 
        
        
        # sentence_X = X.reshape(X.shape[0],-1)
        # padding_rows = np.where(~sentence_X.cpu().numpy().any(axis=1))[0]
        # n_rem_entries= sentence_X.shape[0] - len(padding_rows)
        # #Then, we simple add the padding entries
        # X_sentence_pos = np.concatenate(( np.zeros((len(padding_rows),)), np.array([pos+1 for pos in range(n_rem_entries)])))
        # X_sentence_pos = torch.LongTensor(X_sentence_pos) 
        
        # #an extra [] is necessary since we are getting a float this time
        # Y=torch.FloatTensor([self.y_labels["labels"][hid][self.config["target_label_index"]]])
        
        # if(self.config["loss_function"] !='ll1'):
        #     label_index = self.config["target_label_index"]

        #     target_median_val = self.config["y_score_median_values"][label_index]

        #     Y= (Y>= target_median_val)
        #     #We are doing it for "soft" labeling
        #     #Y = torch.sigmoid(Y - target_median_val)

            
                
        # return X,X_word_pos,X_sentence_pos,Y
    
class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual,acoustic,input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual=visual,
        self.acoustic=acoustic,
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id
    def __str__(self):
        return "inputs_ids:{0},visual:{1},acoustic:{2},input_mask:{3},segment:{4},label_id:{5}".format(self.input_ids,self.visual,self.acoustic,self.input_mask,self.segment_ids,self.label_id)



@ets_bert_ex.config
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
    train_batch_size=2
    learning_rate=5e-5
    num_train_epochs=20.0
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
    h_merge_sent=0
    
    max_num_sentences=0
    Y_size=0
    target_label_index=0
    
    if prototype:
        num_train_epochs=1
        train_batch_size=2
    prot_train=5
    prot_dev=5
    prot_test=2
    
    d_acoustic_in=0
    d_visual_in = 0
    h_audio_lstm = 0
    h_video_lstm = 0
    h_merge_sent = 0
    
    label_median=5.6
        
    

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


@ets_bert_ex.capture
def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode,_config):
    """Loads a data file into a list of `InputBatch`s."""
    #print("label_list:",label_list)
    (all_words,all_visual,all_acoustic,label,segment,id_2_word) = examples
    label_map = {label : i for i, label in enumerate(label_list)}
    #print(len(words),len(visual),len(acoustic),len(label),len(segment),len(id_2_word))
    #print(segment,label)
    
   
    
    
    features = []
    vid_len = len(all_words)
    label=np.array(label)
    for i in range(min(vid_len,_config["max_num_sentences"])): 
    #(ex_index, example) in enumerate(examples):
        words = np.array(all_words[i])
        visual = np.array(all_visual[i])
        acoustic = np.array(all_acoustic[i])
        #(words, visual, acoustic), label, segment = example
        #print(words,label, segment)
        #we will look at acoustic and visual later
        words = " ".join([id_2_word[w] for w in words])
        #print("string word:", words)
        example = InputExample(guid = segment, text_a = words, text_b=None, label=label.item())
        #print(example)
        #In amir's tokenizer, we need to give this invertable=True for it to work properly
        tokens_a,inversions_a = tokenizer.tokenize(example.text_a,invertable=True)
        #print("The new tokenizer:",tokens_a,inversions_a)
        #assert False
        
        #Some words are broken into several tokens. For all those tokens, we are using the same audio+visual features.
        new_visual=[]
        new_audio=[]
        for inv_id in inversions_a:
            new_visual.append(visual[inv_id,:])
            new_audio.append(acoustic[inv_id,:])

        visual = np.array(new_visual) 
        acoustic = np.array(new_audio)
        #print(visual,visual.shape)#47
        #print(acoustic,acoustic.shape)#74
        #TODO:As we do not have a second token, we are keeping it unchanged for now
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
                acoustic = acoustic[:(max_seq_length - 2)]
                visual = visual[:(max_seq_length - 2)]

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
        
        #We ndded to remove some of the acoustic and vis
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        #for now, we will just use zeros for the acourstic and visual info
        audio_zero = np.zeros((1,acoustic.shape[1]))
        acoustic = np.concatenate((audio_zero,acoustic,audio_zero))
        #print("corrected acoustic:",acoustic, acoustic.shape)
        
        visual_zero = np.zeros((1,visual.shape[1]))
        visual = np.concatenate((visual_zero,visual,visual_zero))
        #print("corr visual:",visual,visual.shape)
        
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)
        #print(len(input_ids),len(inversions_a), visual.shape)
        #assert False
        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        
        #Then zero pad the visual and acoustic
        audio_padding = np.zeros((max_seq_length - len(input_ids),acoustic.shape[1]))
        #print("audio pad:",audio_padding, "with:",(max_seq_length - len(input_ids),acoustic.shape[1]))
        acoustic = np.concatenate((acoustic,audio_padding))
        #print("padded acoustic:",acoustic.shape)
        
        video_padding = np.zeros((max_seq_length - len(input_ids),visual.shape[1]))
        visual = np.concatenate((visual,video_padding))
        
        input_ids += padding
        input_mask += padding
        segment_ids += padding
        #print("after meeting:",max_seq_length,acoustic.shape[0],visual.shape[0])

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert acoustic.shape[0] == max_seq_length
        assert visual.shape[0] == max_seq_length

        

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

       

        features.append(
                InputFeatures(input_ids=input_ids,
                              visual=visual,
                              acoustic=acoustic,
                              input_mask=input_mask,
                              segment_ids=segment_ids,
                              label_id=label_id,))
    return features,vid_len


def _truncate_seq_pair(tokens_a, tokens_b, max_length):
    """Truncates a sequence pair in place to the maximum length.""" 
    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_visual,all_acoustic,all_input_mask, all_segment_ids, all_label_ids)

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

@ets_bert_ex.capture
def get_appropriate_dataset(data,tokenizer, output_mode,_config):
    features = convert_examples_to_features(
            data, _config["label_list"],_config["max_seq_length"], tokenizer, output_mode)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
    
    #print("bert_ids:",all_input_ids)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_visual,all_acoustic,all_input_mask, all_segment_ids, all_label_ids)
    return dataset
   

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
@ets_bert_ex.capture
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
    id_2_word_file =  os.path.join(data_path,"ets_word_list.pkl")
        
    word_aligned_facet_sdk=load_pickle(facet_file)
    word_aligned_covarep_sdk=load_pickle(covarep_file)
    word_embedding_idx_sdk=load_pickle(word_vec_file)
    y_labels_sdk = load_pickle(y_labels)
    id_2_word = load_pickle(id_2_word_file)['data']
    #print(id_2_word)
    all_data = (word_aligned_facet_sdk,word_aligned_covarep_sdk,word_embedding_idx_sdk,y_labels_sdk,id_2_word)
    tokenizer = BertTokenizer.from_pretrained(_config["bert_model"], do_lower_case=_config["do_lower_case"])

    
    training_set = ETSDataset(train,_config,all_data,tokenizer)
    dev_set = ETSDataset(dev,_config,all_data,tokenizer)
    test_set = ETSDataset(test,_config,all_data,tokenizer)

    
    #print("dataset init")
    #print("In train dataloader:",_config["train_batch_size"])
    train_dataloader = DataLoader(training_set, batch_size=_config["train_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    dev_dataloader = DataLoader(dev_set, batch_size=_config["dev_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    
    test_dataloader = DataLoader(test_set, batch_size=_config["test_batch_size"],
                        shuffle=_config["shuffle"], num_workers=_config["num_workers"])
    num_train_optimization_steps = int(len(training_set) / _config["train_batch_size"] / _config["gradient_accumulation_steps"]) * _config["num_train_epochs"]
    
    print("num_t:{0}".format(num_train_optimization_steps))

    
    #print("data loader prepared")
    #my_logger.debug(train_X.shape,train_Y.shape,dev_X.shape,dev_Y.shape,test_X.shape,test_Y.shape)
    #data_loader = test_data_loader(train_X,train_Y,_config)
    return train_dataloader,dev_dataloader,test_dataloader,num_train_optimization_steps
    
        
    # with open(os.path.join(_config["dataset_location"],'all_mod_data.pickle'), 'rb') as handle:
    #     all_data = pickle.load(handle)
    # train_data = all_data["train"]
    # dev_data=all_data["dev"]
    # test_data=all_data["test"]
    
    # if(_config["prototype"]):
    #     train_data=train_data[:100]
    #     dev_data=dev_data[:100]
    #     test_data=test_data[:100]     
    
    
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





@ets_bert_ex.capture
def set_random_seed(seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    Parameter:
        seed: It is set in @ex.config and will be passed through variable injection.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

@ets_bert_ex.capture
def prep_for_training(num_train_optimization_steps,_config):
    tokenizer = BertTokenizer.from_pretrained(_config["bert_model"], do_lower_case=_config["do_lower_case"])


    # TODO:Change model here
    model = ETSBertForSequenceClassification.multimodal_from_pretrained(_config["bert_model"],newly_added_config = _config,
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

@ets_bert_ex.capture
def train_epoch(model,train_dataloader,optimizer,_config):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(_config["device"]) for t in batch)
            input_ids, visual,acoustic,input_mask, segment_ids, label_ids,video_lens = batch
            visual = torch.squeeze(visual,2)
            acoustic = torch.squeeze(acoustic,2)
            #print("visual:",visual.shape," acoustic:",acoustic.shape," video_lens:",video_lens.shape)
            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, visual,acoustic,segment_ids, input_mask, labels=None)
            #assert False


            if _config["output_mode"] == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(logits.view(-1, _config["num_labels"]), label_ids.view(-1))
            elif _config["output_mode"] == "regression":
                #print("given:{0},predicted:{1}".format(label_ids,logits))
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

@ets_bert_ex.capture
def eval_epoch(model,dev_dataloader,optimizer,_config):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(_config["device"]) for t in batch)
            input_ids, visual,acoustic,input_mask, segment_ids, label_ids,video_lens = batch
            visual = torch.squeeze(visual,2)
            acoustic = torch.squeeze(acoustic,2)
            #print("visual:",visual.shape," acoustic:",acoustic.shape," video_lens:",video_lens.shape)
            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, visual,acoustic,segment_ids, input_mask, labels=None)
            #print("visual:",visual.shape," acoustic:",acoustic.shape," model type:",type(model))
            #assert False
            # define a new function to compute loss values for both output_modes


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
   
@ets_bert_ex.capture
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
            input_ids, visual,acoustic,input_mask, segment_ids, label_ids,video_lens = batch
            visual = torch.squeeze(visual,2)
            acoustic = torch.squeeze(acoustic,2)
            #print("visual:",visual.shape," acoustic:",acoustic.shape," video_lens:",video_lens.shape)
            # define a new function to compute loss values for both output_modes
            logits = model(input_ids, visual,acoustic,segment_ids, input_mask, labels=None)
            
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
   
@ets_bert_ex.capture
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
    print("Accuracy:{0}, F-1 score:{1}".format(accuracy,f_score))
    
    _run.info['final_result']={'accuracy':accuracy,'mae':mae,'corr':corr,"mult_acc":mult,
             "mult_f_score":f_score,"Confusion Matrix":confusion_matrix_result,
             "Classification Report":classification_report_score}
    return accuracy
            
@ets_bert_ex.capture
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
                else:
                    print("Not an improved dev model")
                    test_score_model(model,test_data_loader)
                    
    #After the entire training is over, save the best model as artifact in the mongodb
    
    
@ets_bert_ex.automain
def main(_config):
    
    set_random_seed(_config["seed"])
    #print(_config["rand_test"],_config["seed"])
    train_data_loader,dev_data_loader,test_data_loader,num_train_optimization_steps = set_up_data_loader()
    
    model,optimizer,tokenizer = prep_for_training(num_train_optimization_steps)

    train(model, train_data_loader,dev_data_loader,test_data_loader,optimizer)
    #assert False

    #TODO:need to fix it
    # test_accuracy = test_score(test_data_loader,criterion)
    # ex.log_scalar("test.accuracy",test_accuracy)
    # results = dict()
    # #I believe that it will try to minimize the rest. Let's see how it plays out
    # results["optimization_target"] = 1 - test_accuracy

    #return results
