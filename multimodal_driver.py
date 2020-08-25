from __future__ import absolute_import, division, print_function

import argparse
import csv
import logging
import os
import random
import pickle
import sys
import numpy as np
from typing import *

from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, f1_score

import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm, trange

from torch.nn import CrossEntropyLoss, L1Loss
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import matthews_corrcoef
from transformers import BertTokenizer
from transformers.optimization import AdamW, WarmupLinearSchedule
from modeling_bert import MAG_BertForSequenceClassification

logger = logging.getLogger(__name__)
from global_configs import ACOUSTIC_DIM, VISUAL_DIM, DEVICE

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", type=str, choices=["mosi", "mosei"], default="mosi")
parser.add_argument("--max_seq_length", type=int, default=50)
parser.add_argument("--train_batch_size", type=int, default=48)
parser.add_argument("--dev_batch_size", type=int, default=128)
parser.add_argument("--test_batch_size", type=int, default=128)
parser.add_argument("--n_epochs", type=int, default=40)
parser.add_argument("--beta_shift", type=float, default=1.0)
parser.add_argument("--bert_model", type=str, default="bert-base-uncased")
parser.add_argument("--learning_rate", type=float, default=1e-5)
parser.add_argument("--gradient_accumulation_step", type=int, default=1)
parser.add_argument("--warmup_proportion", type=float, default=0.1)
parser.add_argument("--seed", type=int, default=-1)


args = parser.parse_args()


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, visual, acoustic, input_mask, segment_ids, label_id):
        self.input_ids = input_ids
        self.visual = visual
        self.acoustic = acoustic
        self.input_mask = input_mask
        self.segment_ids = segment_ids
        self.label_id = label_id


def get_inversion(tokens: List[str], SPIECE_MARKER="▁"):
    """
    Compute inversion indexes for list of tokens.

    Example:
        tokens = ["▁here", "▁is", "▁the", "▁sentence", "▁I", "▁want", "▁em", "bed", "ding", "s", "for"]
        inversions = get_inversion(tokens)
        inversions == [0, 1, 2, 3, 4, 5, 6, 7, 7, 7, 7, 8]

    Args:
        tokens (List[str]): List of word tokens 
        SPIECE_MARKER (str, optional): Special character to beginning of a single "word". Defaults to "▁".

    Returns:
        List[int]: Inversion indexes for each token
    """
    inversion_index = -1
    inversions = []
    for token in tokens:
        if SPIECE_MARKER in token:
            inversion_index += 1
        inversions.append(inversion_index)
    
    return inversions


def convert_to_features(examples, label_list, max_seq_length, tokenizer):

    label_map = {label: i for i, label in enumerate(label_list)}
    with open(os.path.join(args.dataset, "word2id.pickle"), "rb") as handle:
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

        acoustic_zero = np.zeros((1, ACOUSTIC_DIM))
        acoustic = np.concatenate((acoustic_zero, acoustic, acoustic_zero))

        visual_zero = np.zeros((1, VISUAL_DIM))
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


def get_appropriate_dataset(data, tokenizer):
    features = convert_to_features(
        data, args.label_list, args.max_seq_length, tokenizer
    )
    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_input_mask = torch.tensor([f.input_mask for f in features], dtype=torch.long)
    all_segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    all_visual = torch.tensor([f.visual for f in features], dtype=torch.float)
    all_acoustic = torch.tensor([f.acoustic for f in features], dtype=torch.float)
    all_label_ids = torch.tensor([f.label_id for f in features], dtype=torch.float)

    dataset = TensorDataset(
        all_input_ids,
        all_visual,
        all_acoustic,
        all_input_mask,
        all_segment_ids,
        all_label_ids,
    )
    return dataset


def set_up_data_loader():
    with open(os.path.join(args.dataset, "all_mod_data.pickle"), "rb") as handle:
        all_data = pickle.load(handle)

    train_data = all_data["train"]
    dev_data = all_data["dev"]
    test_data = all_data["test"]

    tokenizer = BertTokenizer.from_pretrained(args.bert_model)

    train_dataset = get_appropriate_dataset(train_data, tokenizer)
    dev_dataset = get_appropriate_dataset(dev_data, tokenizer)
    test_dataset = get_appropriate_dataset(test_data, tokenizer)

    num_train_optimization_steps = (
        int(
            len(train_dataset) / args.train_batch_size / args.gradient_accumulation_step
        )
        * args.n_epochs
    )

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.train_batch_size, shuffle=True
    )

    dev_dataloader = DataLoader(
        dev_dataset, batch_size=args.dev_batch_size, shuffle=True
    )

    test_dataloader = DataLoader(
        test_dataset, batch_size=args.test_batch_size, shuffle=True,
    )

    return (
        train_dataloader,
        dev_dataloader,
        test_dataloader,
        num_train_optimization_steps,
    )


def set_random_seed(seed):
    """
    Helper function to seed experiment for reproducibility.
    If -1 is provided as seed, experiment uses random seed from 0~9999

    Args:
        seed (int): integer to be used as seed, use -1 to randomly seed experiment
    """
    if seed == -1:
        seed = random.randint(0, 9999)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.deterministic = True

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def prep_for_training(num_train_optimization_steps):
    tokenizer = BertTokenizer.from_pretrained(args.bert_model)
    model = MAG_BertForSequenceClassification.from_pretrained(
        args.bert_model, num_labels=1, multimodal_config=args,
    )

    model.to(DEVICE)

    # Prepare optimizer
    param_optimizer = list(model.named_parameters())
    no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
    optimizer_grouped_parameters = [
        {
            "params": [
                p for n, p in param_optimizer if not any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.01,
        },
        {
            "params": [
                p for n, p in param_optimizer if any(nd in n for nd in no_decay)
            ],
            "weight_decay": 0.0,
        },
    ]

    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate)
    scheduler = WarmupLinearSchedule(
        optimizer,
        t_total=num_train_optimization_steps,
        warmup_steps=args.warmup_proportion * num_train_optimization_steps,
    )
    return model, optimizer, scheduler, tokenizer


def train_epoch(model, train_dataloader, optimizer, scheduler):
    model.train()
    tr_loss = 0
    nb_tr_examples, nb_tr_steps = 0, 0
    for step, batch in enumerate(tqdm(train_dataloader, desc="Iteration")):
        batch = tuple(t.to(DEVICE) for t in batch)
        input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
        visual = torch.squeeze(visual, 1)
        acoustic = torch.squeeze(acoustic, 1)
        outputs = model(
            input_ids,
            visual,
            acoustic,
            token_type_ids=segment_ids,
            attention_mask=input_mask,
            labels=None,
        )
        logits = outputs[0]
        loss_fct = L1Loss()
        loss = loss_fct(logits.view(-1), label_ids.view(-1))

        if args.gradient_accumulation_step > 1:
            loss = loss / args.gradient_accumulation_step

        loss.backward()

        tr_loss += loss.item()
        nb_tr_examples += input_ids.size(0)
        nb_tr_steps += 1

        if (step + 1) % args.gradient_accumulation_step == 0:
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
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )
            logits = outputs[0]

            loss_fct = L1Loss()
            loss = loss_fct(logits.view(-1), label_ids.view(-1))

            if args.gradient_accumulation_step > 1:
                loss = loss / args.gradient_accumulation_step

            dev_loss += loss.item()
            nb_dev_examples += input_ids.size(0)
            nb_dev_steps += 1

    return dev_loss


def test_epoch(model, data_loader):
    model.eval()
    preds = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(data_loader):
            batch = tuple(t.to(DEVICE) for t in batch)

            input_ids, visual, acoustic, input_mask, segment_ids, label_ids = batch
            visual = torch.squeeze(visual, 1)
            acoustic = torch.squeeze(acoustic, 1)
            outputs = model(
                input_ids,
                visual,
                acoustic,
                token_type_ids=segment_ids,
                attention_mask=input_mask,
                labels=None,
            )

            logits = outputs[0]

            preds.append(logits)
            labels.append(label_ids)

        preds = np.array(preds)
        labels = np.array(labels)

        preds = np.squeeze(logits)
        labels = np.squeeze(labels)

    return preds, labels


def test_score_model(model, test_data_loader, use_zero=False):

    preds, y_test = test_epoch(model, test_data_loader)
    non_zeros = np.array([i for i, e in enumerate(y_test) if e != 0 or use_zero])

    preds = preds[non_zeros]
    y_test = y_test[non_zeros]

    mae = np.mean(np.absolute(preds - y_test))
    corr = np.corrcoef(preds, y_test)[0][1]

    preds = preds >= 0
    f_score = f1_score(y_test, preds, average="weighted")
    accuracy = accuracy_score(y_test, preds)

    print("Accuracy {}".format(accuracy))

    return accuracy, mae, corr, f_score


def train(
    model,
    train_dataloader,
    validation_dataloader,
    test_data_loader,
    optimizer,
    scheduler,
):
    for epoch_i in range(int(args.n_epochs)):
        train_loss = train_epoch(model, train_dataloader, optimizer, scheduler)
        valid_loss = eval_epoch(model, validation_dataloader, optimizer)

        print(
            "epoch:{}, train_loss:{}, valid_loss:{}".format(
                epoch_i, train_loss, valid_loss
            )
        )

        test_accuracy, test_mae, test_corr, test_f_score = test_score_model(
            model, test_data_loader
        )


def main():
    print("Seed: ", args.seed)
    set_random_seed(args.seed)

    (
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        num_train_optimization_steps,
    ) = set_up_data_loader()

    model, optimizer, scheduler = prep_for_training(num_train_optimization_steps)

    train(
        model,
        train_data_loader,
        dev_data_loader,
        test_data_loader,
        optimizer,
        scheduler,
    )


if __name__ == "__main__":
    main()
