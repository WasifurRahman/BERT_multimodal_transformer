#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 14 08:04:26 2019

@author: echowdh2
"""

from driver import ex
import random
import os
import argparse, sys
import pickle
parser=argparse.ArgumentParser()
parser.add_argument('--dataset', help='the dataset you want to work on')
from global_configs import *

dataset_specific_config = {
        #Train:10569,dev:2642,Test:3303
        #t,a,v
        "ETS":{'input_dims':[1,81,35],'embedded_input_dims':[300,81,35], 'mfn_input_dims':[1,81,35,300],'max_num_sentences':20,'max_seq_len':30, 'Y_size':6}
        }

experiment_configs=[
        {'use_text':True,'use_audio':True,'use_video':True,'use_positional_attention':False,"loss_function":"ll1"},#ind 0:T+A+V
        {'use_text':True,'use_audio':True,'use_video':True,'use_positional_attention':True,"loss_function":"ll1"},#1
        {'use_text':True,'use_audio':True,'use_video':True,'use_positional_attention':False,"loss_function":"bce"},#2 ind 0:T+A+V
        {'use_text':True,'use_audio':True,'use_video':True,'use_positional_attention':True,"loss_function":"bce"},#3
        
        {"loss_function":"bce","optim":"transformer"}#4
       
        ]
num_experiments = len(experiment_configs)

#sacred will generate a different random _seed for every experiment
#and we will use that seed to control the randomness emanating from our libraries

if running_as_job_array==True:
    node_index=int(os.environ['SLURM_ARRAY_TASK_ID'])
else:
    node_index=100
#So, we are assuming that there will a folder called /processed_multimodal_data in the parent folder
#of this code. I wanted to keep it inside the .git folder. But git push limits file size to be <=100MB
#and some data files exceeds that size.
all_datasets_location = "../processed_multimodal_data"

num_option_in_Y = 1
emphaisis_on_a_subset=4
run_on_a_seed = 5
run_mfn_trans=6
cur_experiment= emphaisis_on_a_subset

#disabling for now
if conf_inference==True:
    cur_experiment = run_on_a_seed

def run_configs(dataset_location):
    dataset_name = dataset_location[dataset_location.rfind("/")+1:]
    
    if cur_experiment==emphaisis_on_a_subset:
        for relevant_config in [0]:
            for y_index in range(0,num_option_in_Y):
                #print("Running :",str(relevant_config))
                appropriate_config_dict = {**dataset_specific_config[dataset_name],**experiment_configs[relevant_config],"node_index":node_index,
                                                  "prototype":conf_prototype,'dataset_location':dataset_location,"dataset_name":dataset_name,
                                                  "experiment_config_index":relevant_config,'target_label_index':y_index,'inference':conf_inference,\
                                                 "optim":"transformer"}
                r= ex.run(config_updates=appropriate_config_dict)
    elif cur_experiment== run_on_a_seed:
        
        for relevant_config in [4]:
                #chosen_seed=297914073
                #chosen_seed = 26076528
                #for this chosen seed, the config was
                appropriate_config_dict = {**dataset_specific_config[dataset_name],**experiment_configs[relevant_config],"node_index":node_index,
                                                  "prototype":conf_prototype,'dataset_location':dataset_location,"dataset_name":dataset_name,
                                                  "experiment_config_index":relevant_config,'target_label_index':0,'inference':conf_inference,\
                                                   "optim":"transformer"}
                r= ex.run(command_name = "run_on_a_seed_then_run_inference",config_updates=appropriate_config_dict)
    elif cur_experiment == run_mfn_trans:
        for relevant_config in [4]:
                print("inside mfn trans")
                appropriate_config_dict = {**dataset_specific_config[dataset_name],**experiment_configs[relevant_config],"node_index":node_index,
                                                  "prototype":conf_prototype,'dataset_location':dataset_location,"dataset_name":dataset_name,
                                                  "experiment_config_index":relevant_config,'target_label_index':0,'inference':conf_inference\
                                                  
                                                   }
                r= ex.run(config_updates=appropriate_config_dict)
        

            
    
    
    
#run it like ./running_different_configs.py --dataset=ETS
#or python running_different_configs.py --dataset=ETS

if __name__ == '__main__':
    args = parser.parse_args()
    dataset_path = os.path.join(all_datasets_location,args.dataset)
    
    
    if(os.path.isdir(dataset_path)):
        while(True):
            run_configs(dataset_path)
            

            
    else:
        raise NotADirectoryError("Please input the dataset name correctly")
    
    
