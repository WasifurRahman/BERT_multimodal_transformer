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
from global_configs import *

sys.path.insert(0,'./pytorch-transformers')
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

#from pytorch_transformers.file_utils import PYTORCH_PRETRAINED_BERT_CACHE, WEIGHTS_NAME, CONFIG_NAME
from pytorch_transformers.modeling_xlnet import XLNetForSequenceClassification, XLNetConfig, MultimodalXLNetForSequenceClassification
#from pytorch_transformers.tokenization import BertTokenizer
#We are using the tokenization that amir did
from pytorch_transformers.amir_tokenization import XLNetTokenizer

from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

logger = logging.getLogger(__name__)
from sacred import Experiment
import optuna

xlnet_multi_ex = Experiment('xlnet_multimodal_transformer')
from sacred.observers import MongoObserver
from global_configs import *
url_database = conf_url_database
mongo_database_name = conf_mongo_database_name
xlnet_multi_ex.observers.append(MongoObserver.create(url= url_database ,db_name= mongo_database_name))

torch.cuda.empty_cache()

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

    def __init__(self, input_ids, visual,acoustic,input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual=visual,
        self.acoustic=acoustic,
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id



@xlnet_multi_ex.config
def cnf():
    dataset_location=None
    xlnet_model=None
    data_dir=None
    node_index=None
    prototype=None
    dataset_location=None
    dataset_name=None
    task_name=None
    do_train=True
    do_eval=True
    cache_dir=None
    max_seq_length=128
    train_batch_size=32
    learning_rate=5e-5
    num_train_epochs=40.0
    seed=101
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
    best_model_path =  our_model_saving_path +str(node_index) +"xlnet_multi_best_model.chkpt"
    loss_function="ll1"
    save_model=True
    save_mode='best'
    d_acoustic_in=0
    d_visual_in = 0
    h_audio_lstm = 0
    h_video_lstm = 0
    h_merge_sent = 0
    acoustic_in_dim=0
    visual_in_dim=0
    fc1_out=0
    fc1_dropout=0
    hidden_dropout_prob=0
    beta_shift = 0
    AV_index = 1

    if prototype:
        num_train_epochs=2



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


@xlnet_multi_ex.capture
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

        #In amir's tokenizer, we need to give this invertable=True for it to work properly
        tokens_a,inversions_a = tokenizer.tokenize(example.text_a,invertable=True)
        #assert False
        '''
        print('----TOKENS----')
        print(tokens_a)
        print('---INVERSIONS---')
        print(inversions_a)
        '''

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
        tokens = tokens_a + ["[SEP]"] + ["[CLS]"]
        #for now, we will just use zeros for the acourstic and visual info
        audio_zero = np.zeros((1,acoustic.shape[1]))
        acoustic = np.concatenate((acoustic,audio_zero,audio_zero))
        #print("corrected acoustic:",acoustic, acoustic.shape)

        visual_zero = np.zeros((1,visual.shape[1]))
        visual = np.concatenate((visual,visual_zero,visual_zero))
        #print("corr visual:",visual,visual.shape)

        segment_ids = [0] * (len(tokens) - 1) + [2]

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
        acoustic = np.concatenate((audio_padding,acoustic))
        #print("padded acoustic:",acoustic.shape)

        video_padding = np.zeros((max_seq_length - len(input_ids),visual.shape[1]))
        visual = np.concatenate((video_padding,visual))

        input_ids = padding + input_ids
        input_mask = padding + input_mask
        segment_ids = [4] * (max_seq_length - len(segment_ids)) + segment_ids
        #print("after meeting:",max_seq_length,acoustic.shape[0],visual.shape[0])

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert acoustic.shape[0] == max_seq_length
        assert visual.shape[0] == max_seq_length

        '''
        print('---SAMPLE---')
        print('input_ids: ',input_ids)
        print('visual: ',visual.shape,visual)
        print('acoustic: ',acoustic.shape,acoustic)
        print('segment_ids: ',segment_ids)
        print('input_mask: ',input_mask)
        '''

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

@xlnet_multi_ex.capture
def get_appropriate_dataset(data,tokenizer, output_mode,_config):
    features = convert_examples_to_features(
            data, _config["label_list"],_config["max_seq_length"], tokenizer, output_mode)
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)

    #print("xlnet_ids:",all_input_ids)

    if output_mode == "classification":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_visual,all_acoustic,all_input_mask, all_segment_ids, all_label_ids)
    return dataset


@xlnet_multi_ex.capture
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


    tokenizer = XLNetTokenizer.from_pretrained(_config["xlnet_model"])
    output_mode = _config["output_mode"]

    train_dataset = get_appropriate_dataset(train_data,tokenizer, output_mode,_config)
    dev_dataset = get_appropriate_dataset(dev_data,tokenizer, output_mode,_config)
    test_dataset = get_appropriate_dataset(test_data,tokenizer, output_mode,_config)

    #print("train_dataset:",train_dataset)
    #print(len(train_dataset),_config["train_batch_size"],_config["gradient_accumulation_steps"], _config["num_train_epochs"])
    #WE may use it for ETS
    num_train_optimization_steps = int(len(train_dataset) / _config["train_batch_size"] / _config["gradient_accumulation_steps"]) * _config["num_train_epochs"]
    #print("num_tr_opt_st:",num_train_optimization_steps)

    #print("Train len:",len(train_dataset)," dev:",len(dev_dataset)," test:",len(test_dataset))

    train_dataloader = DataLoader(train_dataset, batch_size=_config["train_batch_size"],
                        shuffle=_config["shuffle"], num_workers=1,worker_init_fn=_init_fn)

    dev_dataloader = DataLoader(dev_dataset, batch_size=_config["dev_batch_size"],
                        shuffle=_config["shuffle"], num_workers=1,worker_init_fn=_init_fn)

    test_dataloader = DataLoader(test_dataset, batch_size=_config["test_batch_size"],
                        shuffle=_config["shuffle"], num_workers=1,worker_init_fn=_init_fn)


    #print(train_X.shape,train_Y.shape,dev_X.shape,dev_Y.shape,test_X.shape,test_Y.shape)
    #data_loader = test_data_loader(train_X,train_Y,_config)
    return train_dataloader,dev_dataloader,test_dataloader,num_train_optimization_steps





@xlnet_multi_ex.capture
def set_random_seed(seed):
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    Parameter:
        seed: It is set in @ex.config and will be passed through variable injection.
    """
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

@xlnet_multi_ex.capture
def _init_fn(worker_id,_config):
    random.seed(_config["seed"]+worker_id)
    np.random.seed(_config["seed"]+worker_id)

@xlnet_multi_ex.capture
def prep_for_training(num_train_optimization_steps,_config):
    tokenizer = XLNetTokenizer.from_pretrained(_config["xlnet_model"])


    # TODO:Change model here
    model = MultimodalXLNetForSequenceClassification.multimodal_from_pretrained(_config["xlnet_model"],
              newly_added_config = _config,
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

    optimizer = AdamW(optimizer_grouped_parameters,
                         lr=_config["learning_rate"])
    scheduler = WarmupLinearSchedule(optimizer,t_total=num_train_optimization_steps,
                         warmup_steps=_config["warmup_proportion"] * num_train_optimization_steps)
    return model,optimizer,scheduler,tokenizer

@xlnet_multi_ex.capture
def train_epoch(model,train_dataloader,optimizer,scheduler,_config):
        model.train()
        tr_loss = 0
        nb_tr_examples, nb_tr_steps = 0, 0
        for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
            batch = tuple(t.to(_config["device"]) for t in batch)
            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual,1)
            acoustic = torch.squeeze(acoustic,1)
            #print("text:",input_ids.shape)
            #print("visual:",visual.shape," acoustic:",acoustic.shape," model type:",type(model))
            #assert False
            # define a new function to compute loss values for both output_modes
            outputs = model(input_ids, visual, acoustic, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
            logits = outputs[0]

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
                scheduler.step()
                optimizer.zero_grad()
                #global_step += 1
        return tr_loss

@xlnet_multi_ex.capture
def eval_epoch(model,dev_dataloader,optimizer,_config):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(_config["device"]) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual,1)
            acoustic = torch.squeeze(acoustic,1)
            #print("visual:",visual.shape," acoustic:",acoustic.shape," model type:",type(model))
            #assert False
            # define a new function to compute loss values for both output_modes
            outputs = model(input_ids, visual, acoustic, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
            logits = outputs[0]

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

@xlnet_multi_ex.capture
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

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual,1)
            acoustic = torch.squeeze(acoustic,1)
            #print("visual:",visual.shape," acoustic:",acoustic.shape," model type:",type(model))
            #assert False
            # define a new function to compute loss values for both output_modes
            outputs = model(input_ids, visual, acoustic, token_type_ids=segment_ids, attention_mask=input_mask, labels=None)
            logits = outputs[0]

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

@xlnet_multi_ex.capture
def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))

@xlnet_multi_ex.capture
def test_score_model(model,test_data_loader, _config, _run, exclude_zero=False):

    predictions,y_test = test_epoch(model,test_data_loader)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0.0 or (not exclude_zero)])
    #print("predictions:",predictions,predictions.shape)
    #print("ytest:",y_test,y_test.shape)
    predictions_a7 = np.clip(predictions, a_min=-3., a_max=3.)
    y_test_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    predictions_a5 = np.clip(predictions, a_min=-2., a_max=2.)
    y_test_a5 = np.clip(y_test, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(predictions-y_test))
    #print("mae: ", mae)

    corr = np.corrcoef(predictions,y_test)[0][1]
    #print("corr: ", corr)

    mult_a7 = multiclass_acc(predictions_a7, y_test_a7)
    mult_a5 = multiclass_acc(predictions_a5, y_test_a5)
    #print("mult_acc: ", mult)

    #As we canged the "Y" as probability, now we need to choose yes for >=0.5
    if(_config["loss_function"]=="bce"):
        true_label = (y_test[non_zeros] >= 0.5)
    elif(_config["loss_function"]=="ll1"):
        true_label = (y_test[non_zeros] >= 0)

    predicted_label = (predictions[non_zeros] >= 0)

    f_score = f1_score(true_label,predicted_label, average='weighted')

    confusion_matrix_result = confusion_matrix(true_label, predicted_label)
    classification_report_score = classification_report(true_label, predicted_label, digits=5)
    accuracy = accuracy_score(true_label, predicted_label)

    print("Accuracy ",accuracy )

    r={'accuracy':accuracy,'mae':mae,'corr':corr,"mult_a5":mult_a5,"mult_a7":mult_a7,
             "mult_f_score":f_score,"Confusion Matrix":confusion_matrix_result,
             "Classification Report":classification_report_score}

    if exclude_zero:
        if 'final_result' in _run.info.keys():
            _run.info['final_result'].append(r)
        else:
            _run.info['final_result']=[r]

    return accuracy, mae, corr, mult_a5, mult_a7, f_score

@xlnet_multi_ex.capture
def train(model, train_dataloader, validation_dataloader,test_data_loader,optimizer,scheduler,_config,_run):
    ''' Start training '''
    model_path = _config["best_model_path"]
    best_test_acc = 0.0

    valid_losses = []
    for epoch_i in range(int(_config["num_train_epochs"])):

        #print('[ Epoch', epoch_i, ']')


        train_loss = train_epoch(model,train_dataloader,optimizer,scheduler)
        #print("\nepoch:{},train_loss:{}".format(epoch_i,train_loss))
        _run.log_scalar("training.loss", train_loss, epoch_i)


        valid_loss = eval_epoch(model, validation_dataloader,optimizer)
        _run.log_scalar("dev.loss", valid_loss, epoch_i)

        '''
        trial.report(valid_loss,epoch_i)
        if trial.should_prune():
            raise optuna.structs.TrialPruned()
        '''

        valid_losses.append(valid_loss)
        print("\nepoch:{},train_loss:{}, valid_loss:{}".format(epoch_i,train_loss,valid_loss))

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            '_config': _config,
            'epoch': epoch_i
        }

        test_accuracy, test_mae, test_corr, test_mult_a5, test_mult_a7, test_f_score = test_score_model(model,test_data_loader)
        zero_test_acurracy, _, _, _, _, zero_test_f1 = test_score_model(model,test_data_loader,exclude_zero=True)

        _run.log_scalar("test_per_epoch.acc", test_accuracy, epoch_i)
        _run.log_scalar("test_per_epoch.mae", test_mae, epoch_i)
        _run.log_scalar("test_per_epoch.corr", test_corr, epoch_i)
        _run.log_scalar("test_per_epoch.mult_a5", test_mult_a5, epoch_i)
        _run.log_scalar("test_per_epoch.mult_a7", test_mult_a7, epoch_i)
        _run.log_scalar("test_per_epoch.f_score", test_f_score, epoch_i)

        _run.log_scalar("test_per_epoch.zero_acc", zero_test_acurracy, epoch_i)
        _run.log_scalar("test_per_epcoh.zero_f1", zero_test_f1, epoch_i)

        if _config["save_model"]:
            # if _config["save_mode"] == 'all':
            #     model_name = _config["save_model"] + '_accu_{accu:3.3f}.chkpt'.format(accu=100*valid_loss)
            #     torch.save(checkpoint, model_name)
            if _config["save_mode"] == 'best':
                #print(_run.experiment_info)
                if valid_loss <= min(valid_losses):
                    torch.save(checkpoint, model_path)
                    print('    - [Info] The checkpoint file has been updated.')
                    _run.info['best_val_loss'] = valid_loss
                if test_accuracy >= best_test_acc:
                    _run.info['best_test_acc'] = test_accuracy
                    best_test_acc = test_accuracy
    #After the entire training is over, save the best model as artifact in the mongodb


@xlnet_multi_ex.automain
def main(_config):

    print("Seed: ",_config["seed"])
    set_random_seed(_config["seed"])
    #print(_config["rand_test"],_config["seed"])
    train_data_loader,dev_data_loader,test_data_loader,num_train_optimization_steps = set_up_data_loader()

    model,optimizer,scheduler,tokenizer = prep_for_training(num_train_optimization_steps)

    train(model,train_data_loader,dev_data_loader,test_data_loader,optimizer,scheduler)

    # assert False

    #TODO:need to fix it
    # test_accuracy = test_score(test_data_loader,criterion)
    # ex.log_scalar("test.accuracy",test_accuracy)
    # results = dict()
    # #I believe that it will try to minimize the rest. Let's see how it plays out
    # results["optimization_target"] = 1 - test_accuracy

    #return results
