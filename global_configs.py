#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 08:34:31 2019

@author: echowdh2
"""

running_as_job_array = False
conf_prototype=False #use during development, not in the funal version
conf_inference=False ##do not bother right now
conf_url_database = 'bhg0017' #where is the database?
all_datasets_location = "/home/slee232/scratch/processed_multimodal_data/"#The place where you are storing the data
CACHE_DIR="/home/slee232/scratch/processed_multimodal_data/MRPC/model_cache"#make sure that it is avalid directory
our_model_saving_path = "/home/slee232/scratch/saved_models_from_projects/bert_transformer/"#make sure that it is valid


if(conf_prototype == True):
    conf_mongo_database_name = 'prototype'
else:#the real databse you want to explore and report the results on
    #conf_mongo_database_name = 'mfn_multi_single'
    #conf_mongo_database_name = 'bert_late_fusion'
    #conf_mongo_database_name = 'bert_late_fusion_no_shift'
    #conf_mongo_database_name = 'bert_gated_shift'
    #conf_mongo_database_name = 'bert_text_only'
    conf_mongo_database_name = 'multi_gated_shift'
    #conf_mongo_database_name = 'multi_gated_shift_final'
