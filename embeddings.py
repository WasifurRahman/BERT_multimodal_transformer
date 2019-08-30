from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import pickle
import sys
import torch
import re
from global_configs import *

import pytorch_transformers
from pytorch_transformers import BertModel, BertTokenizer, XLNetModel, XLNetTokenizer

torch.cuda.synchronize()
tokenizer = XLNetTokenizer.from_pretrained("xlnet-base-cased")
model = XLNetModel.from_pretrained("xlnet-base-cased")
device = torch.device('cuda')
model.to(device)

def return_unk():
    return 0

max_seq_length = 35
with open(os.path.join('/scratch/slee232/processed_multimodal_data/mosi/','word2id.pickle'), 'rb') as handle:
    word_2_id = pickle.load(handle)
id_2_word = { id_:word for (word,id_) in word_2_id.items()}

for partition in ['train','dev','test']:
    with open(os.path.join('/scratch/slee232/processed_multimodal_data/mosi/','all_mod_data.pickle'), 'rb') as handle:
        examples = pickle.load(handle)[partition]
    features = []

    for example in examples:
        (words, visual, acoustic), label, segment = example
        words = " ".join([id_2_word[w] for w in words])
        tokens_a = tokenizer.tokenize(words)
        if len(tokens_a) > max_seq_length - 2:
            tokens_a = tokens_a[:(max_seq_length - 2)]
        tokens =  tokens_a + ["[SEP]"] + ["[CLS]"]
        segment_ids = [0] * (len(tokens)-1) + [2]
        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        input_mask = [1] * len(input_ids)

        padding = [0] * (max_seq_length - len(input_ids))
        input_ids = padding + input_ids
        input_mask = padding + input_mask
        segment_ids = [4] * (max_seq_length - len(segment_ids)) + segment_ids

        input_ids = torch.tensor(input_ids).unsqueeze(0).to(device)
        input_mask = torch.tensor(input_mask).unsqueeze(0).to(device)
        segment_ids = torch.tensor(segment_ids).unsqueeze(0).to(device)

        model.eval()
        with torch.no_grad():
            outputs = model(input_ids,token_type_ids=segment_ids, attention_mask=input_mask)
        embedding = outputs[0]
        features.append(embedding.cpu().detach().numpy())
    with open('/scratch/slee232//processed_multimodal_data/mosei/'+'mosi_'+partition+'_xlnet.pickle','wb') as out:
        pickle.dump(features,out)
