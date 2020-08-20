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

from torch.nn import CrossEntropyLoss, MSELoss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from pytorch_transformers.amir_tokenization import BertTokenizer

from pytorch_transformers.optimization import AdamW, WarmupLinearSchedule

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
        print("guid:{0},text_a:{1},text_b:{2},label:{3}".format(
            self.guid, self.text_a, self.text_b, self.label))


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
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


def convert_examples_to_features(examples, label_list, max_seq_length,
                                 tokenizer, output_mode, config):
    """Loads a data file into a list of `InputBatch`s."""
    # print("label_list:",label_list)

    label_map = {label: i for i, label in enumerate(label_list)}
    with open(os.path.join(config["dataset_location"], 'word2id.pickle'), 'rb') as handle:
        word_2_id = pickle.load(handle)
    id_2_word = {id_: word for (word, id_) in word_2_id.items()}
    # print(id_2_word)

    features = []
    for (ex_index, example) in enumerate(examples):

        (words, visual, acoustic), label, segment = example
        #print(words,label, segment)
        # we will look at acoustic and visual later
        words = " ".join([id_2_word[w] for w in words])
        #print("string word:", words)
        example = InputExample(guid=segment, text_a=words,
                               text_b=None, label=label.item())
        # print(example)

        tokens_a, _ = tokenizer.tokenize(example.text_a, invertable=True)

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
        tokens = ["[CLS]"] + tokens_a + ["[SEP]"]
        segment_ids = [0] * len(tokens)

        if tokens_b:
            tokens += tokens_b + ["[SEP]"]
            segment_ids += [1] * (len(tokens_b) + 1)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        input_mask = [1] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding = [0] * (max_seq_length - len(input_ids))
        input_ids += padding
        input_mask += padding
        segment_ids += padding

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        if output_mode == "classification":
            label_id = label_map[example.label]
        elif output_mode == "regression":
            label_id = float(example.label)
        else:
            raise KeyError(output_mode)

        features.append(
            InputFeatures(input_ids=input_ids,
                          input_mask=input_mask,
                          segment_ids=segment_ids,
                          label_id=label_id))
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


def get_appropriate_dataset(data, tokenizer, output_mode, config):
    features = convert_examples_to_features(
        data, config["label_list"], config["max_seq_length"], tokenizer, output_mode)
    all_input_ids = torch.tensor(
        [f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor(
        [f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor(
        [f.segment_ids for f in features], dtype=torch.long)

    # print("bert_ids:",all_input_ids)

    if output_mode == "classification":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.long)
    elif output_mode == "regression":
        all_label_ids = torch.tensor(
            [f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(all_input_ids, all_input_mask,
                            all_segment_ids, all_label_ids)
    return dataset


def set_up_data_loader(config):

    with open(os.path.join(config["dataset_location"], 'all_mod_data.pickle'), 'rb') as handle:
        all_data = pickle.load(handle)
    train_data = all_data["train"]
    dev_data = all_data["dev"]
    test_data = all_data["test"]

    if(config["prototype"]):
        train_data = train_data[:100]
        dev_data = dev_data[:100]
        test_data = test_data[:100]

    tokenizer = BertTokenizer.from_pretrained(
        config["bert_model"], do_lower_case=config["do_lower_case"])
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
    """
    This function controls the randomness by setting seed in all the libraries we will use.
    Parameter:
        seed: It is set in @ex.config and will be passed through variable injection.
    """
    print('INSIDE: ', seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True


def prep_for_training(num_train_optimization_steps, config):
    tokenizer = BertTokenizer.from_pretrained(
        config["bert_model"], do_lower_case=config["do_lower_case"])

    # TODO:Change model here
    model = BertForSequenceClassification.from_pretrained(config["bert_model"],
                                                          cache_dir=config["cache_dir"],
                                                          num_labels=config["num_labels"])

    model.to(config["device"])

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
        batch = tuple(t.to(config["device"]) for t in batch)
        input_ids, input_mask, segment_ids, label_ids = batch

        # define a new function to compute loss values for both output_modes
        outputs = model(input_ids, token_type_ids=segment_ids,
                        attention_mask=input_mask, labels=None)
        logits = outputs[0]

        if config["output_mode"] == "classification":
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(
                logits.view(-1, config["num_labels"]), label_ids.view(-1))

        elif config["output_mode"] == "regression":
            loss_fct = MSELoss()
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
            #global_step += 1

    return tr_loss


@bert_ex.capture
def eval_epoch(model, dev_dataloader, optimizer, config):
    model.eval()
    dev_loss = 0
    nb_dev_examples, nb_dev_steps = 0, 0
    with torch.no_grad():
        for step, batch in enumerate(tqdm(dev_dataloader, desc="Iteration")):
            batch = tuple(t.to(config["device"]) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # define a new function to compute loss values for both output_modes
            outputs = model(input_ids, token_type_ids=segment_ids,
                            attention_mask=input_mask, labels=None)
            logits = outputs[0]

            if config["output_mode"] == "classification":
                loss_fct = CrossEntropyLoss()
                loss = loss_fct(
                    logits.view(-1, config["num_labels"]), label_ids.view(-1))
            elif config["output_mode"] == "regression":
                loss_fct = MSELoss()
                loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if config["gradient_accumulation_steps"] > 1:
                loss = loss / config["gradient_accumulation_steps"]

            dev_loss += loss.item()
            nb_dev_examples += input_ids.size(0)
            nb_dev_steps += 1

    return dev_loss


def test_epoch(model, data_loader, config):
    ''' Epoch operation in evaluation phase '''

    model.eval()
    eval_loss = 0.0
    nb_eval_steps = 0
    preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader, mininterval=2, desc='  - (Validation)   ', leave=False):
            batch = tuple(t.to(config["device"]) for t in batch)
            input_ids, input_mask, segment_ids, label_ids = batch

            # define a new function to compute loss values for both output_modes
            outputs = model(input_ids, token_type_ids=segment_ids,
                            attention_mask=input_mask, labels=None)
            logits = outputs[0]

            # create eval loss and other metric required by the task
            if config["output_mode"] == "classification":
                loss_fct = CrossEntropyLoss()
                tmp_eval_loss = loss_fct(
                    logits.view(-1, num_labels), label_ids.view(-1))
            elif config["output_mode"] == "regression":
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
        all_labels = all_labels[0]

        if config["output_mode"] == "classification":
            preds = np.argmax(preds, axis=1)

        elif config["output_mode"] == "regression":
            preds = np.squeeze(preds)
            all_labels = np.squeeze(all_labels)

    return preds, all_labels


def test_score_model(model, test_data_loader, config, run, exclude_zero=False):

    predictions, y_test = test_epoch(model, test_data_loader)
    non_zeros = np.array([i for i, e in enumerate(
        y_test) if e != 0.0 or (not exclude_zero)])

    predictions_a7 = np.clip(predictions, a_min=-3., a_max=3.)
    y_test_a7 = np.clip(y_test, a_min=-3., a_max=3.)
    predictions_a5 = np.clip(predictions, a_min=-2., a_max=2.)
    y_test_a5 = np.clip(y_test, a_min=-2., a_max=2.)

    mae = np.mean(np.absolute(predictions - y_test))

    corr = np.corrcoef(predictions, y_test)[0][1]

    mult_a7 = multiclass_acc(predictions_a7, y_test_a7)
    mult_a5 = multiclass_acc(predictions_a5, y_test_a5)

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
        if 'final_result' in run.info.keys():
            run.info['final_result'].append(r)
        else:
            run.info['final_result'] = [r]

    return accuracy, mae, corr, mult_a5, mult_a7, f_score


@bert_ex.capture
def train(model, train_dataloader, validation_dataloader, test_data_loader, optimizer, scheduler, config, run):
    ''' Start training '''
    model_path = config["best_model_path"]
    best_test_acc = 0.0

    valid_losses = []
    for epoch_i in range(int(config["num_train_epochs"])):
        #print('[ Epoch', epoch_i, ']')

        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        # print("\nepoch:{},train_loss:{}".format(epoch_i,train_loss))
        run.log_scalar("training.loss", train_loss, epoch_i)

        valid_loss = eval_epoch(model, validation_dataloader, optimizer)
        run.log_scalar("dev.loss", valid_loss, epoch_i)

        valid_losses.append(valid_loss)
        print("\nepoch:{},train_loss:{}, valid_loss:{}".format(
            epoch_i, train_loss, valid_loss))

        model_state_dict = model.state_dict()
        checkpoint = {
            'model': model_state_dict,
            'config': config,
            'epoch': epoch_i}
        test_accuracy, test_mae, test_corr, test_mult_a5, test_mult_a7, test_f_score = test_score_model(
            model, test_data_loader)
        zero_test_acurracy, _, _, _, _, zero_test_f1 = test_score_model(
            model, test_data_loader, exclude_zero=True)

        run.log_scalar("test_per_epoch.acc", test_accuracy, epoch_i)
        run.log_scalar("test_per_epoch.mae", test_mae, epoch_i)
        run.log_scalar("test_per_epoch.corr", test_corr, epoch_i)
        run.log_scalar("test_per_epoch.mult_a5", test_mult_a5, epoch_i)
        run.log_scalar("test_per_epoch.mult_a7", test_mult_a7, epoch_i)
        run.log_scalar("test_per_epoch.f_score", test_f_score, epoch_i)

        run.log_scalar("test_per_epoch.zero_acc", zero_test_acurracy, epoch_i)
        run.log_scalar("test_per_epcoh.zero_f1", zero_test_f1, epoch_i)


def main(config):
    print("Seed: ", config["seed"])
    set_random_seed(config["seed"])
    train_data_loader, dev_data_loader, test_data_loader, num_train_optimization_steps = set_up_data_loader()

    model, optimizer, scheduler, tokenizer = prep_for_training(
        num_train_optimization_steps)

    train(model, train_data_loader, dev_data_loader,
          test_data_loader, optimizer, scheduler)
