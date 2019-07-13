#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 08:34:31 2019

@author: echowdh2
"""

running_as_job_array = False
conf_prototype=False
conf_inference=False
conf_url_database = 'bhc0087:27017'
all_datasets_location = "/home/echowdh2/Research_work/processed_multimodal_data"
model_saving_head = "/scratch/echowdh2/saved_models_from_projects/bert_transformer/"


if(conf_prototype == True):
    conf_mongo_database_name = 'prototype'
else:
    #conf_mongo_database_name = 'mfn_multi_single'
    #conf_mongo_database_name = 'bert_late_fusion'
    #conf_mongo_database_name = 'bert_late_fusion_no_shift'
    #conf_mongo_database_name = 'bert_gated_shift'
    #conf_mongo_database_name = 'bert_text_only'
    #conf_mongo_database_name = 'multi_gated_shift'
    conf_mongo_database_name = 'untrained_multi_gated'
    #conf_mongo_database_name = 'multi_gated_shift_final'
#conf_mongo_database_name = 'mfn_text_only'
#run by:node_modules/.bin/omniboard -m  bhg0039:27017:last_stand
