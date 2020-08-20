from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import pickle
import sys
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

from torch.nn import CrossEntropyLoss, L1Loss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer
from transformers.optimization import AdamW, WarmupLinearSchedule

logger = logging.getLogger(__name__)
from global_configs import *


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual,
        self.acoustic = acoustic,
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def multi_collate(batch):
    '''
    Collate functions assume batch = [Dataset[i] for i in index_set]
    '''
    # for later use we sort the batch in descending order of length
    batch = sorted(batch, key=lambda x: x[0][0].shape[0], reverse=True)

    # get the data out of the batch - use pad sequence util functions from PyTorch to pad things
    labels = torch.cat([torch.from_numpy(sample[1])
                        for sample in batch], dim=0)
    sentences = pad_sequence([torch.LongTensor(sample[0][0])
                              for sample in batch], padding_value=PAD)
    visual = pad_sequence([torch.FloatTensor(sample[0][1])
                           for sample in batch])
    acoustic = pad_sequence([torch.FloatTensor(sample[0][2])
                             for sample in batch])

    # lengths are useful later in using RNNs
    lengths = torch.LongTensor([sample[0][0].shape[0] for sample in batch])
    return sentences, visual, acoustic, labels, lengths


def get_inversion(tokens, SPIECE_MARKER="▁"):
    inversion_index = -1
    inversions = []
    for token in tokens:
        if SPIECE_MARKER in token:
            inversion_index += 1
        inversions.append(inversion_index)
    return inversions


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode):

    label_map = {label: i for i, label in enumerate(label_list)}
    with open(os.path.join(config["dataset_location"], 'word2id.pickle'), 'rb') as handle:
        word_2_id = pickle.load(handle)
    id_2_word = {id_: word for (word, id_) in word_2_id.items()}

    features = []
    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label, segment = example
        words = " ".join([id_2_word[w] for w in words])

        tokens = tokenizer.tokenize(words)
        inversions = get_inversion(tokens)

        new_visual = []
        new_audio = []

        for inv_id in inversions:
            new_visual.append(visual[inv_id, :])
            new_audio.append(acoustic[inv_id, :])

        visual = np.array(new_visual)
        acoustic = np.array(new_audio)

        tokens = ["[CLS]"] + tokens + ["[SEP]"]

        acoustic_zero = np.zeros((1, MOSI_ACOUSTIC_DIM))
        acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))

        visual_zero = np.zeros((1, MOSI_VISUAL_DIM))
        visual = np.concatenate((visual_zero, visual, visual_zero))

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        segment_ids = [0] * len(input_ids)
        input_mask = [1] * len(input_ids)

        acoustic_padding = np.zeros(
            (args.max_seq_length - len(input_ids), acoustic.shape[1])
        )
        acoustic = np.concatenate((acoustic, acoustic_padding))

        visual_padding = np.zeros(
            (args.max_seq_length - len(input_ids), visual.shape[1])
        )
        visual = np.concatenate((visual, visual_padding))

        padding = [0] * (args.max_seq_length - len(input_ids))

        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == args.max_seq_length
        assert len(input_mask) == args.max_seq_length
        assert len(segment_ids) == args.max_seq_length
        assert acoustic.shape[0] == args.max_seq_length
        assert visual.shape[0] == args.max_seq_length

        label_id = float(example.label)

        placeholder_sentiment = {"punchline": {"distribution": np.zeros(64)}}

        features.append(
            InputFeatures(
                input_ids=input_ids,
                input_mask=input_mask,
                segment_ids=segment_ids,
                visual=visual,
                acoustic=acoustic,
                label_id=label_id,
                sentiment=placeholder_sentiment,
            )
        )
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


def get_appropriate_dataset(data, tokenizer, output_mode):
    features = convert_examples_to_features(
        data, config["label_list"], config["max_seq_length"], tokenizer, output_mode)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor(
        [f.acoustic for f in features], dtype=torch.float)
    if output_mode == "classification":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_visual, all_acoustic,
                            all_input_mask, all_segment_ids, all_label_ids)
    return dataset


def set_up_data_loader(config):
    with open(os.path.join(config["dataset_location"], 'all_mod_data.pickle'), 'rb') as handle:
        all_data = pickle.load(handle)

    train_data = all_data["train"]
    dev_data = all_data["dev"]
    test_data = all_data["test"]

    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
    output_mode = config["output_mode"]

    train_dataset = get_appropriate_dataset(
        train_data, tokenizer, output_mode, config)
    dev_dataset = get_appropriate_dataset(
        dev_data, tokenizer, output_mode, config)
    test_dataset = get_appropriate_dataset(
        test_data, tokenizer, output_mode, config)

    num_train_optimization_steps = int(len(
        train_dataset) / config["train_batch_size"] / config["gradient_accumulation_steps"]) * config["num_train_epochs"]

    train_dataloader = DataLoader(train_dataset, batch_size=config["train_batch_size"],
                                  shuffle=config["shuffle"], num_workers=1, worker_init_fn=_init_fn)

    dev_dataloader = DataLoader(dev_dataset, batch_size=config["dev_batch_size"],
                                shuffle=config["shuffle"], num_workers=1, worker_init_fn=_init_fn)

    test_dataloader = DataLoader(test_dataset, batch_size=config["test_batch_size"],
                                 shuffle=config["shuffle"], num_workers=1, worker_init_fn=_init_fn)

    return train_dataloader, dev_dataloader, test_dataloader, num_train_optimization_steps


def set_random_seed(seed):
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training(num_train_optimization_steps, config):
    tokenizer = BertTokenizer.from_pretrained(config["bert_model"])
    model = MultimodalBertForSequenceClassification.multimodal_from_pretrained(config["bert_model"], newly_addedconfig=config,
                                                                               cache_dir=config["cache_dir"],
                                                                               num_labels=config["num_labels"])

    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(
            nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(
            nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]

    optimizer = AdamW(optimizer_grouped_parameters,
                      lr=config["learning_rate"])
    scheduler = WarmupLinearSchedule(optimizer, t_total=num_train_optimization_steps,
                                     warmup_steps=config["warmup_proportion"] * num_train_optimization_steps)
    return model, optimizer, scheduler, tokenizer


def train_epoch(model, train_dataloader, optimizer, scheduler, config):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        outputs = model(input_ids, visual, acoustic, token_type_ids=segment_ids,
                        attention_mask=input_mask, labels=None)
        logits = outputs[0]

        if config["output_mode"] == "classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, config["num_labels"]), label_ids.view(-1))
        elif config["output_mode"] == "regression":
            loss_fct = L1Loss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

        if config["gradient_accumulation_steps"] > 1:
            loss = loss / config["gradient_accumulation_steps"]

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        if (step + 1) % config["gradient_accumulation_steps"] == 0:
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

    return tr_loss


def eval_epoch(model, dev_dataloader, optimizer):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(input_ids, visual, acoustic, token_type_ids=segment_ids,
                            attention_mask=input_mask, labels=None)
            logits = outputs[0]

            if config["output_mode"] == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, config["num_labels"]), label_ids.view(-1))
            elif config["output_mode"] == "regression":
                loss_fct = L1Loss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if config["gradient_accumulation_steps"] > 1:
                loss = loss / config["gradient_accumulation_steps"]

            dev_loss += loss.item()
            nb_dev_examples += input_ids.size(0)
            nb_dev_steps += 1

    return dev_loss


def test_epoch(model, data_loader):
    model.eval()
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    all_labels = []

    with torch.no_grad():

        for batch in tqdm(data_loader, mininterval=2, desc='  - (Validation)   ', leave=False):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(input_ids, visual, acoustic, token_type_ids=segment_ids,
                            attention_mask=input_mask, labels=None)
            logits = outputs[0]

            # create eval loss and other metric required by the task
            if config["output_mode"] == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(
                    logits.view(-1, num_labels), label_ids.view(-1))
            elif config["output_mode"] == "regression":
                loss_fct = L1Loss()
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
        all_labels = all_labels[0]

        if config["output_mode"] == "classification":
            preds = np.argmax(preds, axis=1)
        elif config["output_mode"] == "regression":
            preds = np.squeeze(preds)
            all_labels = np.squeeze(all_labels)

    return preds, all_labels


def multiclass_acc(preds, truths):
    """
    Compute the multiclass accuracy w.r.t. groundtruth
    :param preds: Float array representing the predictions, dimension (N,)
    :param truths: Float/int array representing the groundtruth classes, dimension (N,)
    :return: Classification accuracy
    """
    return np.sum(np.round(preds) == np.round(truths)) / float(len(truths))


def test_score_model(model, test_data_loader):

    predictions, y_test = test_epoch(model, test_data_loader)
    non_zeros = np.array([i for i, e in enumerate(
        y_test) if e != 0 or (not exclude_zero)])
    predictions_a7 = np.clip(predictions, a_min=-3., a_max=3.)
    y_test_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    predictions_a5 = np.clip(predictions, a_min=-2., a_max=2.)
    y_test_a5 = np.clip(y_test, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(predictions - y_test))
    #print("mae: ", mae)

    corr = np.corrcoef(predictions, y_test)[0][1]
    #print("corr: ", corr)

    mult_a7 = multiclass_acc(predictions_a7, y_test_a7)
    mult_a5 = multiclass_acc(predictions_a5, y_test_a5)
    #print("mult_acc: ", mult)

    # As we canged the "Y" as probability, now we need to choose yes for >=0.5
    if(config["loss_function"] == "bce"):
        true_label = (y_test[non_zeros] >= 0.5)
    elif(config["loss_function"] == "ll1"):
        true_label = (y_test[non_zeros] >= 0)

    predicted_label = (predictions[non_zeros] >= 0)

    f_score = f1_score(true_label, predicted_label, average='weighted')

    confusion_matrix_result = confusion_matrix(true_label, predicted_label)
    classification_report_score = classification_report(
        true_label, predicted_label, digits=5)
    accuracy = accuracy_score(true_label, predicted_label)

    print("Accuracy ", accuracy)

    r = {'accuracy': accuracy, 'mae': mae, 'corr': corr, "mult_a5": mult_a5, "mult_a7": mult_a7,
         "mult_f_score": f_score, "Confusion Matrix": confusion_matrix_result,
         "Classification Report": classification_report_score}

    if exclude_zero:
        if 'final_result' in _run.info.keys():
            _run.info['final_result'].append(r)
        else:
            _run.info['final_result'] = [r]

    return accuracy, mae, corr, mult_a5, mult_a7, f_score


def train(model, train_dataloader, validation_dataloader, test_data_loader, optimizer, scheduler, config):
    for epoch_i in range(int(config["num_train_epochs"])):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader, optimizer)

        print("epoch:{},train_loss:{}, valid_loss:{}".format(
            epoch_i, train_loss, valid_loss))

        model_state_dict = model.state_dict()

        test_accuracy, test_mae, test_corr, test_mult_a5, test_mult_a7, test_f_score = test_score_model(
            model, test_data_loader)


def main():
    print("Seed: ", config["seed"])
    set_random_seed(config["seed"])

    train_data_loader, dev_data_loader, test_data_loader, num_train_optimization_steps = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(
        num_train_optimization_steps)

    train(model, train_data_loader, dev_data_loader,
          test_data_loader, optimizer, scheduler)
