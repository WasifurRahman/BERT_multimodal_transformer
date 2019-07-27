import sys
#sys.path.insert(0, '/home/echowdh2/Research_work/general_purpose_code/downloaded_softwares/keras')

import numpy as np
#seed = 123
seed=123475
np.random.seed(seed)
import random
import torch
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable, grad
from torch.optim.lr_scheduler import ReduceLROnPlateau

import h5py
import time
import data_loader as loader
from collections import defaultdict, OrderedDict
import argparse
import pickle as pickle
import time
import json, os, ast, h5py

#from keras.models import Model
#from keras.layers import Input
#from keras.layers.embeddings import Embedding

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import sys



# def get_data(args,config):
#     tr_split = 2.0/3                        # fixed. 62 training & validation, 31 test
#     val_split = 0.1514                      # fixed. 52 training 10 validation
#     use_pretrained_word_embedding = True    # fixed. use glove 300d
#     embedding_vecor_length = 300            # fixed. use glove 300d
#     # 115                   # fixed for MOSI. The max length of a segment in MOSI dataset is 114 
#     max_segment_len = config['seqlength']
#     end_to_end = True                       # fixed

#     word2ix = loader.load_word2ix()
#     word_embedding = [loader.load_word_embedding()] if use_pretrained_word_embedding else None
#     train, valid, test = loader.load_word_level_features(max_segment_len, tr_split)

#     ix2word = inv_map = {v: k for k, v in word2ix.iteritems()}
#     print(len(word2ix))
#     print(len(ix2word))
#     print(word_embedding[0].shape)

#     feature_str = ''
#     if args.feature_selection:
#         with open('/media/bighdd5/Paul/mosi/fs_mask.pkl') as f:
#             [covarep_ix, facet_ix] = pickle.load(f)
#         facet_train = train['facet'][:,:,facet_ix]
#         facet_valid = valid['facet'][:,:,facet_ix]
#         facet_test = test['facet'][:,:,facet_ix]
#         covarep_train = train['covarep'][:,:,covarep_ix]
#         covarep_valid = valid['covarep'][:,:,covarep_ix]
#         covarep_test = test['covarep'][:,:,covarep_ix]
#         feature_str = '_t'+str(embedding_vecor_length) + '_c'+str(covarep_test.shape[2]) + '_f'+str(facet_test.shape[2])
#     else:
#         facet_train = train['facet']
#         facet_valid = valid['facet']
#         covarep_train = train['covarep'][:,:,1:35]
#         covarep_valid = valid['covarep'][:,:,1:35]
#         facet_test = test['facet']
#         covarep_test = test['covarep'][:,:,1:35]

#     text_train = train['text']
#     text_valid = valid['text']
#     text_test = test['text']
#     y_train = train['label']
#     y_valid = valid['label']
#     y_test = test['label']

#     lengths_train = train['lengths']
#     lengths_valid = valid['lengths']
#     lengths_test = test['lengths']

#     #f = h5py.File("out/mosi_lengths_test.hdf5", "w")
#     #f.create_dataset('d1',data=lengths_test)
#     #f.close()
#     #assert False

#     facet_train_max = np.max(np.max(np.abs(facet_train ), axis =0),axis=0)
#     facet_train_max[facet_train_max==0] = 1
#     #covarep_train_max =  np.max(np.max(np.abs(covarep_train), axis =0),axis=0)
#     #covarep_train_max[covarep_train_max==0] = 1

#     facet_train = facet_train / facet_train_max
#     facet_valid = facet_valid / facet_train_max
#     #covarep_train = covarep_train / covarep_train_max
#     facet_test = facet_test / facet_train_max
#     #covarep_test = covarep_test / covarep_train_max

#     text_input = Input(shape=(max_segment_len,), dtype='int32', name='text_input')
#     text_eb_layer = Embedding(word_embedding[0].shape[0], embedding_vecor_length, input_length=max_segment_len, weights=word_embedding, name = 'text_eb_layer', trainable=False)(text_input)
#     model = Model(text_input, text_eb_layer)
#     text_train_emb = model.predict(text_train)
#     print(text_train_emb.shape)      # n x seq x 300
#     print(covarep_train.shape)      # n x seq x 5/34
#     print(facet_train.shape)         # n x seq x 20/43
#     X_train = np.concatenate((text_train_emb, covarep_train, facet_train), axis=2)

#     text_valid_emb = model.predict(text_valid)
#     print(text_valid_emb.shape)      # n x seq x 300
#     print(covarep_valid.shape)       # n x seq x 5/34
#     print(facet_valid.shape)         # n x seq x 20/43
#     X_valid = np.concatenate((text_valid_emb, covarep_valid, facet_valid), axis=2)

#     text_test_emb = model.predict(text_test)
#     print(text_test_emb.shape)      # n x seq x 300
#     print(covarep_test.shape)       # n x seq x 5/34
#     print(facet_test.shape)         # n x seq x 20/43
#     X_test = np.concatenate((text_test_emb, covarep_test, facet_test), axis=2)

#     return X_train, y_train, X_valid, y_valid, X_test, y_test

def load_bert_data(data_location,max_seq_len=-1):
    print(data_location,max_seq_len)
    pkl_file = open(data_location, 'rb')
    data = pickle.load(pkl_file)
    pkl_file.close()
    
    X_train = data['X_train'][:,:max_seq_len,:]
    y_train=data['y_train']
    X_valid = data['X_valid'][:,:max_seq_len,:]
    y_valid=data['y_valid']
    X_test=data['X_test'][:,:max_seq_len,:]
    y_test=data['y_test']
    #{'X_train':X_train,'y_train':y_train,'X_valid':X_valid,'y_valid':y_valid,'X_test':X_test,'y_test':y_test}
    print(X_train.shape,y_train.shape,X_valid.shape,y_valid.shape,X_test.shape,y_test.shape)
    return (X_train,y_train,X_valid,y_valid,X_test,y_test)


# def load_saved_data():
#     h5f = h5py.File('data/X_train.h5','r')
#     X_train = h5f['data'][:]
#     h5f.close()
#     h5f = h5py.File('data/y_train.h5','r')
#     y_train = h5f['data'][:]
#     h5f.close()
#     h5f = h5py.File('data/X_valid.h5','r')
#     X_valid = h5f['data'][:]
#     h5f.close()
#     h5f = h5py.File('data/y_valid.h5','r')
#     y_valid = h5f['data'][:]
#     h5f.close()
#     h5f = h5py.File('data/X_test.h5','r')
#     X_test = h5f['data'][:]
#     h5f.close()
#     h5f = h5py.File('data/y_test.h5','r')
#     y_test = h5f['data'][:]
#     h5f.close()
#     return X_train, y_train, X_valid, y_valid, X_test, y_test

# parser = argparse.ArgumentParser(description='')
# parser.add_argument('--config', default='configs/mosi.json', type=str)
# parser.add_argument('--type', default='mgddm', type=str)    # d, gd, m1, m3
# parser.add_argument('--fusion', default='mfn', type=str)    # ef, tf, mv, marn, mfn
# parser.add_argument('-s', '--feature_selection', default=1, type=int, choices=[0,1], help='whether to use feature_selection')

# args = parser.parse_args()
# config = json.load(open(args.config), object_pairs_hook=OrderedDict)

# class EFLSTM(nn.Module):
#     def __init__(self, d, h, output_dim, dropout): #, n_layers, bidirectional, dropout):
#         super(EFLSTM, self).__init__()
#         self.h = h
#         self.lstm = nn.LSTMCell(d, h)
#         self.fc1 = nn.Linear(h, h)
#         self.fc2 = nn.Linear(h, output_dim)
#         self.dropout = nn.Dropout(dropout)
        
#     def forward(self, x):
#         # x is t x n x d
#         t = x.shape[0]
#         n = x.shape[1]
#         self.hx = torch.zeros(n, self.h).cuda()
#         self.cx = torch.zeros(n, self.h).cuda()
#         all_hs = []
#         all_cs = []
#         for i in range(t):
#             self.hx, self.cx = self.lstm(x[i], (self.hx, self.cx))
#             all_hs.append(self.hx)
#             all_cs.append(self.cx)
#         # last hidden layer last_hs is n x h
#         last_hs = all_hs[-1]
#         output = F.relu(self.fc1(last_hs))
#         output = self.dropout(output)
#         output = self.fc2(output)
#         return output

class MFN(nn.Module):
    def __init__(self,config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig):
        super(MFN, self).__init__()
        [self.d_l,self.d_a,self.d_v] = config["input_dims"]
        [self.dh_l,self.dh_a,self.dh_v] = config["h_dims"]
        total_h_dim = self.dh_l+self.dh_a+self.dh_v
        self.mem_dim = config["memsize"]
        window_dim = config["windowsize"]
        output_dim = 1
        attInShape = total_h_dim*window_dim
        gammaInShape = attInShape+self.mem_dim
        final_out = total_h_dim+self.mem_dim
        h_att1 = NN1Config["shapes"]
        h_att2 = NN2Config["shapes"]
        h_gamma1 = gamma1Config["shapes"]
        h_gamma2 = gamma2Config["shapes"]
        h_out = outConfig["shapes"]
        att1_dropout = NN1Config["drop"]
        att2_dropout = NN2Config["drop"]
        gamma1_dropout = gamma1Config["drop"]
        gamma2_dropout = gamma2Config["drop"]
        out_dropout = outConfig["drop"]

        self.lstm_l = nn.LSTMCell(self.d_l, self.dh_l)
        self.lstm_a = nn.LSTMCell(self.d_a, self.dh_a)
        self.lstm_v = nn.LSTMCell(self.d_v, self.dh_v)

        self.att1_fc1 = nn.Linear(attInShape, h_att1)
        self.att1_fc2 = nn.Linear(h_att1, attInShape)
        self.att1_dropout = nn.Dropout(att1_dropout)

        self.att2_fc1 = nn.Linear(attInShape, h_att2)
        self.att2_fc2 = nn.Linear(h_att2, self.mem_dim)
        self.att2_dropout = nn.Dropout(att2_dropout)

        self.gamma1_fc1 = nn.Linear(gammaInShape, h_gamma1)
        self.gamma1_fc2 = nn.Linear(h_gamma1, self.mem_dim)
        self.gamma1_dropout = nn.Dropout(gamma1_dropout)

        self.gamma2_fc1 = nn.Linear(gammaInShape, h_gamma2)
        self.gamma2_fc2 = nn.Linear(h_gamma2, self.mem_dim)
        self.gamma2_dropout = nn.Dropout(gamma2_dropout)

        self.out_fc1 = nn.Linear(final_out, h_out)
        self.out_fc2 = nn.Linear(h_out, output_dim)
        self.out_dropout = nn.Dropout(out_dropout)
        
    def forward(self,x):
        x_l = x[:,:,:self.d_l]
        x_a = x[:,:,self.d_l:self.d_l+self.d_a]
        x_v = x[:,:,self.d_l+self.d_a:]
        # x is t x n x d
        n = x.shape[1]
        t = x.shape[0]
        self.h_l = torch.zeros(n, self.dh_l).cuda()
        self.h_a = torch.zeros(n, self.dh_a).cuda()
        self.h_v = torch.zeros(n, self.dh_v).cuda()
        self.c_l = torch.zeros(n, self.dh_l).cuda()
        self.c_a = torch.zeros(n, self.dh_a).cuda()
        self.c_v = torch.zeros(n, self.dh_v).cuda()
        self.mem = torch.zeros(n, self.mem_dim).cuda()
        all_h_ls = []
        all_h_as = []
        all_h_vs = []
        all_c_ls = []
        all_c_as = []
        all_c_vs = []
        all_mems = []
        for i in range(t):
            # prev time step
            prev_c_l = self.c_l
            prev_c_a = self.c_a
            prev_c_v = self.c_v
            # curr time step
            new_h_l, new_c_l = self.lstm_l(x_l[i], (self.h_l, self.c_l))
            new_h_a, new_c_a = self.lstm_a(x_a[i], (self.h_a, self.c_a))
            new_h_v, new_c_v = self.lstm_v(x_v[i], (self.h_v, self.c_v))
            # concatenate
            prev_cs = torch.cat([prev_c_l,prev_c_a,prev_c_v], dim=1)
            new_cs = torch.cat([new_c_l,new_c_a,new_c_v], dim=1)
            cStar = torch.cat([prev_cs,new_cs], dim=1)
            attention = F.softmax(self.att1_fc2(self.att1_dropout(F.relu(self.att1_fc1(cStar)))),dim=1)
            attended = attention*cStar
            #IN the paper, it is uHat
            cHat = F.tanh(self.att2_fc2(self.att2_dropout(F.relu(self.att2_fc1(attended)))))
            both = torch.cat([attended,self.mem], dim=1)
            gamma1 = F.sigmoid(self.gamma1_fc2(self.gamma1_dropout(F.relu(self.gamma1_fc1(both)))))
            gamma2 = F.sigmoid(self.gamma2_fc2(self.gamma2_dropout(F.relu(self.gamma2_fc1(both)))))
            self.mem = gamma1*self.mem + gamma2*cHat
            all_mems.append(self.mem)
            # update
            self.h_l, self.c_l = new_h_l, new_c_l
            self.h_a, self.c_a = new_h_a, new_c_a
            self.h_v, self.c_v = new_h_v, new_c_v
            all_h_ls.append(self.h_l)
            all_h_as.append(self.h_a)
            all_h_vs.append(self.h_v)
            all_c_ls.append(self.c_l)
            all_c_as.append(self.c_a)
            all_c_vs.append(self.c_v)

        # last hidden layer last_hs is n x h
        last_h_l = all_h_ls[-1]
        last_h_a = all_h_as[-1]
        last_h_v = all_h_vs[-1]
        last_mem = all_mems[-1]
        last_hs = torch.cat([last_h_l,last_h_a,last_h_v,last_mem], dim=1)
        output = self.out_fc2(self.out_dropout(F.relu(self.out_fc1(last_hs))))
        return output

# def train_ef(X_train, y_train, X_valid, y_valid, X_test, y_test, config):
#     p = np.random.permutation(X_train.shape[0])
#     X_train = X_train[p]
#     y_train = y_train[p]

#     X_train = X_train.swapaxes(0,1)
#     X_valid = X_valid.swapaxes(0,1)
#     X_test = X_test.swapaxes(0,1)

#     d = X_train.shape[2]
#     h = config["h"]
#     t = X_train.shape[0]
#     output_dim = 1
#     dropout = config["drop"]

#     model = EFLSTM(d,h,output_dim,dropout)
    
#     optimizer = optim.Adam(model.parameters(),lr=config["lr"])
#     #optimizer = optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"])

#     # optimizer = optim.SGD([
#     #                 {'params':model.lstm_l.parameters(), 'lr':config["lr"]},
#     #                 {'params':model.classifier.parameters(), 'lr':config["lr"]}
#     #             ], momentum=0.9)

#     criterion = nn.L1Loss()
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model = model.to(device)
#     criterion = criterion.to(device)
#     scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)

#     def train(model, batchsize, X_train, y_train, optimizer, criterion):
#         epoch_loss = 0
#         model.train()
#         total_n = X_train.shape[1]
#         num_batches = total_n / batchsize
#         for batch in xrange(num_batches):
#             start = batch*batchsize
#             end = (batch+1)*batchsize
#             optimizer.zero_grad()
#             batch_X = torch.Tensor(X_train[:,start:end]).cuda()
#             batch_y = torch.Tensor(y_train[start:end]).cuda()
#             predictions = model.forward(batch_X).squeeze(1)
#             loss = criterion(predictions, batch_y)
#             loss.backward()
#             optimizer.step()
#             epoch_loss += loss.item()
#         return epoch_loss / num_batches

#     def evaluate(model, X_valid, y_valid, criterion):
#         epoch_loss = 0
#         model.eval()
#         with torch.no_grad():
#             batch_X = torch.Tensor(X_valid).cuda()
#             batch_y = torch.Tensor(y_valid).cuda()
#             predictions = model.forward(batch_X).squeeze(1)
#             epoch_loss = criterion(predictions, batch_y).item()
#         return epoch_loss

#     def predict(model, X_test):
#         epoch_loss = 0
#         model.eval()
#         with torch.no_grad():
#             batch_X = torch.Tensor(X_test).cuda()
#             predictions = model.forward(batch_X).squeeze(1)
#             predictions = predictions.cpu().data.numpy()
#         return predictions

#     # timing
#     start_time = time.time()
#     predictions = predict(model, X_test)
#     print(predictions.shape)
#     print(predictions)
#     end_time = time.time()
#     print(end_time-start_time)
#     assert False

#     best_valid = 999999.0
#     rand = random.randint(0,100000)
#     for epoch in range(config["num_epochs"]):
#         train_loss = train(model, config["batchsize"], X_train, y_train, optimizer, criterion)
#         valid_loss = evaluate(model, X_valid, y_valid, criterion)
#         scheduler.step(valid_loss)
#         if valid_loss <= best_valid:
#             # save model
#             best_valid = valid_loss
#             print(epoch, train_loss, valid_loss, 'saving model')
#             torch.save(model, 'res_mfn2/mfn_%d.pt' %rand)
#         else:
#             print(epoch, train_loss, valid_loss)

#     model = torch.load('res_mfn2/mfn_%d.pt' %rand)

#     predictions = predict(model, X_test)
#     mae = np.mean(np.absolute(predictions-y_test))
#     print("mae: ", mae)
#     corr = np.corrcoef(predictions,y_test)[0][1]
#     print("corr: ", corr)
#     mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
#     print("mult_acc: ", mult)
#     f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
#     print("mult f_score: ", f_score)
#     true_label = (y_test >= 0)
#     predicted_label = (predictions >= 0)
#     print("Confusion Matrix :")
#     print(confusion_matrix(true_label, predicted_label))
#     print("Classification Report :")
#     print(classification_report(true_label, predicted_label, digits=5))
#     print("Accuracy ", accuracy_score(true_label, predicted_label))
#     sys.stdout.flush()

def train_mfn(X_train, y_train, X_valid, y_valid, X_test, y_test, configs):
    p = np.random.permutation(X_train.shape[0])
    X_train = X_train[p]
    y_train = y_train[p]

    X_train = X_train.swapaxes(0,1)
    X_valid = X_valid.swapaxes(0,1)
    X_test = X_test.swapaxes(0,1)

    d = X_train.shape[2]
    h = 128
    t = X_train.shape[0]
    output_dim = 1
    dropout = 0.5

    [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig] = configs

    #model = EFLSTM(d,h,output_dim,dropout)
    model = MFN(config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig)

    optimizer = optim.Adam(model.parameters(),lr=config["lr"])
    #optimizer = optim.SGD(model.parameters(),lr=config["lr"],momentum=config["momentum"])

    # optimizer = optim.SGD([
    #                 {'params':model.lstm_l.parameters(), 'lr':config["lr"]},
    #                 {'params':model.classifier.parameters(), 'lr':config["lr"]}
    #             ], momentum=0.9)

    criterion = nn.L1Loss()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = criterion.to(device)
    scheduler = ReduceLROnPlateau(optimizer,mode='min',patience=100,factor=0.5,verbose=True)

    def train(model, batchsize, X_train, y_train, optimizer, criterion):
        epoch_loss = 0
        model.train()
        total_n = X_train.shape[1]
        num_batches = total_n / batchsize
        for batch in range(int(num_batches)):
            start = batch*batchsize
            end = (batch+1)*batchsize
            optimizer.zero_grad()
            batch_X = torch.Tensor(X_train[:,start:end]).cuda()
            batch_y = torch.Tensor(y_train[start:end]).cuda()
            predictions = model.forward(batch_X).squeeze(1)
            loss = criterion(predictions, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        return epoch_loss / num_batches

    def evaluate(model, X_valid, y_valid, criterion):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_valid).cuda()
            batch_y = torch.Tensor(y_valid).cuda()
            predictions = model.forward(batch_X).squeeze(1)
            epoch_loss = criterion(predictions, batch_y).item()
        return epoch_loss

    def predict(model, X_test):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_test).cuda()
            predictions = model.forward(batch_X).squeeze(1)
            predictions = predictions.cpu().data.numpy()
        return predictions
    def final_score(model,X_test,y_test,where):
        predictions = predict(model, X_test)
        mae = np.mean(np.absolute(predictions-y_test))
        print("mae: ", mae)
        corr = np.corrcoef(predictions,y_test)[0][1]
        print("corr: ", corr)
        mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
        print("mult_acc: ", mult)
        f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
        print("mult f_score: ", f_score)
        true_label = (y_test >= 0)
        predicted_label = (predictions >= 0)
        print("Confusion Matrix :")
        print(confusion_matrix(true_label, predicted_label))
        print("Classification Report :")
        print(classification_report(true_label, predicted_label, digits=5))
        acc =  accuracy_score(true_label, predicted_label)
        print("Accuracy ",acc)
        
        sys.stdout.flush()
        accuracy_out_file.write(where +":" + str(acc)+"\n")
            

    best_valid = 999999.0
    rand = random.randint(0,100000)
    for epoch in range(config["num_epochs"]):
        train_loss = train(model, config["batchsize"], X_train, y_train, optimizer, criterion)
        valid_loss = evaluate(model, X_valid, y_valid, criterion)
        scheduler.step(valid_loss)
        if valid_loss <= best_valid:
            # save model
            best_valid = valid_loss
            print(epoch, train_loss, valid_loss, 'saving model')
            torch.save(model, 'temp_models/mfn_%d.pt' %rand)
            final_score(model,X_test,y_test,"inter")
        else:
            print(epoch, train_loss, valid_loss)

    print('model number is:', rand)
    model = torch.load('temp_models/mfn_%d.pt' %rand)
    final_score(model,X_test,y_test,"final")

    

def test(X_test, y_test, metric):
    X_test = X_test.swapaxes(0,1)
    def predict(model, X_test):
        epoch_loss = 0
        model.eval()
        with torch.no_grad():
            batch_X = torch.Tensor(X_test).cuda()
            predictions = model.forward(batch_X).squeeze(1)
            predictions = predictions.cpu().data.numpy()
        return predictions
    if metric == 'mae':
        model = torch.load('best/mfn_mae.pt')
    if metric == 'acc':
        model = torch.load('best/mfn_acc.pt')
    model = model.cpu().cuda()
    
    predictions = predict(model, X_test)
    print(predictions.shape)
    print(y_test.shape)
    mae = np.mean(np.absolute(predictions-y_test))
    print("mae: ", mae)
    corr = np.corrcoef(predictions,y_test)[0][1]
    print("corr: ", corr)
    mult = round(sum(np.round(predictions)==np.round(y_test))/float(len(y_test)),5)
    print("mult_acc: ", mult)
    f_score = round(f1_score(np.round(predictions),np.round(y_test),average='weighted'),5)
    print("mult f_score: ", f_score)
    true_label = (y_test >= 0)
    predicted_label = (predictions >= 0)
    print("Confusion Matrix :")
    print(confusion_matrix(true_label, predicted_label))
    print("Classification Report :")
    print(classification_report(true_label, predicted_label, digits=5))
    print("Accuracy ", accuracy_score(true_label, predicted_label))
    sys.stdout.flush()

local = False

# if local:
#     X_train, y_train, X_valid, y_valid, X_test, y_test = get_data(args,config)

#     h5f = h5py.File('data/X_train.h5', 'w')
#     h5f.create_dataset('data', data=X_train)
#     h5f = h5py.File('data/y_train.h5', 'w')
#     h5f.create_dataset('data', data=y_train)
#     h5f = h5py.File('data/X_valid.h5', 'w')
#     h5f.create_dataset('data', data=X_valid)
#     h5f = h5py.File('data/y_valid.h5', 'w')
#     h5f.create_dataset('data', data=y_valid)
#     h5f = h5py.File('data/X_test.h5', 'w')
#     h5f.create_dataset('data', data=X_test)
#     h5f = h5py.File('data/y_test.h5', 'w')
#     h5f.create_dataset('data', data=y_test)

#     sys.stdout.flush()
data_location = "/gpfs/fs1/home/echowdh2/Research_work/processed_multimodal_data/bert_mosi/bert_mosi.pkl"
X_train, y_train, X_valid, y_valid, X_test, y_test = load_bert_data(data_location,max_seq_len=35)

#test(X_test, y_test, 'mae')
#test(X_test, y_test, 'acc')
#assert False

#config = dict()
#config["batchsize"] = 32
#config["num_epochs"] = 100
#config["lr"] = 0.01
#config["h"] = 128
#config["drop"] = 0.5
#train_ef(X_train, y_train, X_valid, y_valid, X_test, y_test, config)
#assert False
while True:
    # mae 0.993 [{'input_dims': [300, 5, 20], 'batchsize': 128, 'memsize': 128, 
    #'windowsize': 2, 'lr': 0.01, 'num_epochs': 100, 'h_dims': [88, 48, 16], 'momentum': 0.9}, 
    #{'shapes': 128, 'drop': 0.0}, {'shapes': 64, 'drop': 0.2}, 
    #{'shapes': 256, 'drop': 0.0}, {'shapes': 64, 'drop': 0.2}, 
    #{'shapes': 64, 'drop': 0.5}]

    # acc 77.0 [{'input_dims': [300, 5, 20], 'batchsize': 128, 'memsize': 400, 
    #'windowsize': 2, 'lr': 0.005, 'num_epochs': 100, 'h_dims': [64, 8, 80], 'momentum': 0.9}, 
    #{'shapes': 128, 'drop': 0.5}, {'shapes': 128, 'drop': 0.2}, 
    #{'shapes': 128, 'drop': 0.5}, {'shapes': 128, 'drop': 0.5}, 
    #{'shapes': 256, 'drop': 0.5}]
    accuracy_out_file=open("mfn_bert_acc.txt","a")

    config = dict()
    config["input_dims"] = [768,74,47]
    hl = random.choice([32,64,88,128,156,256,512,768])
    ha = random.choice([32,48,64,80])
    hv = random.choice([32,48,64,80])
    config["h_dims"] = [hl,ha,hv]
    config["memsize"] = random.choice([64,128,256,300,400])
    config["windowsize"] = 2
    config["batchsize"] = random.choice([32,64,128,256])
    config["num_epochs"] = 100
    config["lr"] = random.choice([0.001,0.002,0.005,0.008,0.01])
    config["momentum"] = random.choice([0.1,0.3,0.5,0.6,0.8,0.9])
    NN1Config = dict()
    NN1Config["shapes"] = random.choice([32,64,128,256])
    NN1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    NN2Config = dict()
    NN2Config["shapes"] = random.choice([32,64,128,256])
    NN2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    gamma1Config = dict()
    gamma1Config["shapes"] = random.choice([32,64,128,256])
    gamma1Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    gamma2Config = dict()
    gamma2Config["shapes"] = random.choice([32,64,128,256])
    gamma2Config["drop"] = random.choice([0.0,0.2,0.5,0.7])
    outConfig = dict()
    outConfig["shapes"] = random.choice([32,64,128,256])
    outConfig["drop"] = random.choice([0.0,0.2,0.5,0.7])
    configs = [config,NN1Config,NN2Config,gamma1Config,gamma2Config,outConfig]
    print(configs)
    #assert False
    train_mfn(X_train, y_train, X_valid, y_valid, X_test, y_test, configs)
    
    accuracy_out_file.close()


