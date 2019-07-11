#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 08:34:31 2019

@author: echowdh2
"""

running_as_job_array = False
conf_prototype=True
conf_inference=True
conf_url_database = 'bhc0087:27017'

if(conf_prototype == True):
    conf_mongo_database_name = 'prototype'
else:
    #conf_mongo_database_name = 'mfn_multi_single'
    #conf_mongo_database_name = 'ets_multi'
    conf_mongo_database_name = 'ets_reu'
    

#conf_mongo_database_name = 'mfn_text_only'
#run by:node_modules/.bin/omniboard -m  bhg0039:27017:last_stand
