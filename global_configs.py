#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 08:34:31 2019

@author: echowdh2
"""

running_as_job_array = False
conf_prototype=False#use during development, not in the funal version
conf_inference=False##do not bother right now
conf_url_database = 'bhc0085:27017'#where is the database?
all_datasets_location = "/scratch/mhasan8/processed_multimodal_data"#The place where you are storing the data
CACHE_DIR="/scratch/mhasan8/processed_multimodal_data/MRPC/model_cache"#make sure that it is avalid directory
our_model_saving_path = "/scratch/mhasan8/saved_models_from_projects/bert_transformer/"#make sure that it is valid


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
#conf_mongo_database_name = 'mfn_text_only'
#run by:node_modules/.bin/omniboard -m  bhg0039:27017:last_stand
