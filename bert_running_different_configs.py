#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  7 13:06:48 2019

@author: echowdh2
"""

#from staged_multimodal_transformer_driver import ex
import os
import argparse, sys
from global_configs import *
from sacred import Experiment
import numpy as np

skeleton_ex = Experiment('launcher')
from sample_example_driver import main_ex
#works on text only
from bert_mosi_driver import bert_ex
from bert_multi_mosi_driver import bert_multi_ex
from ets_bert_driver import ets_bert_ex



parser=argparse.ArgumentParser()
parser.add_argument('--dataset', help='the dataset you want to work on')

dataset_specific_config = {
        "mosi":{'input_modalities_sizes':[300,5,20],'output_mode':'regression','label_list':[None],'dev_batch_size':229,'test_batch_size':685,'d_acoustic_in':74,'d_visual_in':47},
        "ETS":{'input_modalities_sizes':[1,81,35],'output_mode':'regression','label_list':[None],'dev_batch_size':229,'test_batch_size':685,'d_acoustic_in':81,'d_visual_in':35,'max_num_sentences':20,'max_seq_length':30, 'Y_size':6,'target_label_index':0},
        "iemocap":{'text_indices':(0,300),'audio_indices':(300,374),'video_indices':(374,409),'max_seq_len':21},
        "mmmo":{'text_indices':(0,300),'audio_indices':(300,374),'video_indices':(374,409),'max_seq_len':21},
        "moud":{'text_indices':(0,300),'audio_indices':(300,374),'video_indices':(374,409),'max_seq_len':21},
        "pom":{'text_indices':(0,300),'audio_indices':(300,343),'video_indices':(343,386),'max_seq_len':21},
        "youtube":{'text_indices':(0,300),'audio_indices':(300,374),'video_indices':(374,409),'max_seq_len':21},
        "MRPC":{}
        
        }

#sacred will generate a different random _seed for every experiment
#and we will use that seed to control the randomness emanating from our libraries
if(running_as_job_array == True):
 node_index=int(os.environ['SLURM_ARRAY_TASK_ID'])
else: 
    node_index=50

#So, we are assuming that there will a folder called /processed_multimodal_data in the parent folder
#of this code. I wanted to keep it inside the .git folder. But git push limits file size to be <=100MB
#and some data files exceeds that size.
all_datasets_location = "/scratch/mhasan8/processed_multimodal_data"

@skeleton_ex.config
def sk_config():
    dataset_location=None

@skeleton_ex.command
def initiate_main_experiment(_config):
    #config_to_init_main=_config["skeleton_init_config"]
    dataset_location = _config["dataset_location"]
    dataset_name = dataset_location[dataset_location.rfind("/")+1:]
    main_init_configs = {**dataset_specific_config[dataset_name],"node_index":node_index,

                              "prototype":conf_prototype,'dataset_location':dataset_location,"dataset_name":dataset_name}
    
    GLUE_DIR="/scratch/mhasan8/processed_multimodal_data/"
    CACHE_DIR="/scratch/mhasan8/processed_multimodal_data/MRPC/model_cache"
    TASK_NAME=dataset_name
    main_init_configs["task_name"] = TASK_NAME
    main_init_configs["do_train"]  = True
    main_init_configs["do_eval"]  =True
    main_init_configs["do_lower_case"] =True
    main_init_configs["data_dir"]  = GLUE_DIR + "/" +TASK_NAME 
    main_init_configs["cache_dir"] = CACHE_DIR 
    main_init_configs["bert_model"] = "bert-base-uncased"
    main_init_configs["max_seq_length"]  = 35 #TODO:May be shortened
    main_init_configs["train_batch_size"] =  32 
    main_init_configs["learning_rate"]  = np.random.choice([2e-5,2e-6,2e-4]) 
    main_init_configs["h_merge_sent"] = 768
    main_init_configs["acoustic_in_dim"] = 74
    main_init_configs["visual_in_dim"] = 47
    main_init_configs["hidden_dropout_prob"]=np.random.choice([0.1,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8])
    main_init_configs["beta_shift"]=np.random.choice([0.25,0.3,0.35,0.4,0.45,0.5,0.6,0.7,0.8,0.9,1,1.5,1.8,2,2.2,2.5,2.8,3,3.2,3.4,3.6,3.8,4,4.5,4.8,5,5.5,5.8,6,6.2,6.4,7,8,10,12,14,16,20,22,24,28,30,32,40,44,50,60,70,80,100])
    
    # main_init_configs["hidden_dropout_prob"]=0.45
    # main_init_configs["beta_shift"]=3
  
    
    
    main_init_configs["h_audio_lstm"] = np.random.choice([16,32,48,64,56,128])
    main_init_configs["h_video_lstm"] = np.random.choice([16,32,48,64,56])
    
    main_init_configs["fc1_out"] = np.random.choice([32,64,128,512])
    main_init_configs["fc1_dropout"] = np.random.choice([0.1,0.2,0.3,0.4,0.45,0.5,0.6])
    
    main_init_configs["num_train_epochs"] =  60
    #commenting out temporarily
    main_init_configs["output_dir"] =  "/tmp/"+TASK_NAME
    #fix the seed beforehand
    main_init_configs["seed"] = _config["seed"]

    
    #print("inherited this configs:",main_init_configs,main_init_configs.keys())
    #result = bert_ex.run(command_name="main",config_updates=main_init_configs)
    #return
    if dataset_name=="mosi":
        result = bert_multi_ex.run(command_name="main",config_updates=main_init_configs)
    elif dataset_name=="ETS": 
        result = ets_bert_ex.run(command_name="main",config_updates=main_init_configs)

    #must use seed for the main exp

def return_unk():
    return 0

def run_a_config(dataset_location):
    #print(dataset_location)
    #dataset_name = dataset_location[dataset_location.rfind("/")+1:]
    # appropriate_configs = {**dataset_specific_config[dataset_name],"node_index":node_index,

    #                           "prototype":conf_prototype,'dataset_location':dataset_location,"dataset_name":dataset_name}
    # skeleton_init_config = {"skeleton_init_config":appropriate_configs}
    #print(appropriate_config_dict)
    skeleton_ex.run(command_name= 'initiate_main_experiment',config_updates={'dataset_location':dataset_location})
    #r = ex.run(named_configs=['search_space'],config_updates={"node_index":node_index,"prototype":True})
    
    
#run it like ./bert_running_different_configs.py --dataset=mosi
#run: ./bert_running_different_configs.py --dataset=ETS    
if __name__ == '__main__':
    args = parser.parse_args()
    dataset_path = os.path.join(all_datasets_location,args.dataset)
    if(os.path.isdir(dataset_path)):
        while(True):
            run_a_config(dataset_path)
        
    else:
        raise NotADirectoryError("Please input the dataset name correctly")
    
    # subfolders = [f.path for f in os.scandir(all_datasets_location) if f.is_dir() ]  
    
    # for s_folder in subfolders:
        
        
    #     dataset_name = s_folder[s_folder.rfind("/")+1:]
    #     if dataset_name == args.dataset:
    #         run_a_config(s_folder)
    #         break
       
            
    
   