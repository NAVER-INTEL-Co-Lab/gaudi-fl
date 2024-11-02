import warnings
warnings.filterwarnings("ignore", message="torch.distributed.reduce_op is deprecated")

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoConfig, AutoModelForSequenceClassification

import copy
import numpy as np
import random
from tqdm import trange
import habana_frameworks.torch.core as htcore
import logging

from utils.options import args_parser
from utils.dataset import load_data, partition_data
from utils.test import test_model
from src.server_opt import server_avg
from src.edge_opt import EdgeOpt
from src.plugin import apply_iid_approximation, balanced_sampling

loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
for logger in loggers:
    if "transformers" in logger.name.lower():
        logger.setLevel(logging.ERROR)

def init_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def initialize_global_model(args):
    config = AutoConfig.from_pretrained(args.model_name_or_path, num_labels=4)
    config.num_labels = 5
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)
    return model

def early_stop_check(test_acc, best_acc, patience_counter, args):
    if test_acc < best_acc + args.delta:
        patience_counter += 1
    else:
        best_acc = test_acc
        patience_counter = 0

    early_stop = patience_counter == args.patience
    return best_acc, patience_counter, early_stop

def log_performance(iter, test_acc, test_loss, best_acc, args):
    if args.plugin:
        writer.add_scalar(f'testacc/{args.model_name_or_path}_{args.frac}_{args.method}_{args.dirichlet_alpha}_plugin_{args.plugin}_{args.syn}', test_acc, iter)
        writer.add_scalar(f'bestacc/{args.model_name_or_path}_{args.frac}_{args.method}_{args.dirichlet_alpha}_plugin_{args.plugin}_{args.syn}', best_acc, iter)
        writer.add_scalar(f'testloss/{args.model_name_or_path}_{args.frac}_{args.method}_{args.dirichlet_alpha}_plugin_{args.plugin}_{args.syn}', test_loss, iter)
    else:
        writer.add_scalar(f'testacc/{args.model_name_or_path}_{args.frac}_{args.method}_{args.dirichlet_alpha}_plugin_{args.plugin}_unused', test_acc, iter)
        writer.add_scalar(f'bestacc/{args.model_name_or_path}_{args.frac}_{args.method}_{args.dirichlet_alpha}_plugin_{args.plugin}_unused', best_acc, iter)
        writer.add_scalar(f'testloss/{args.model_name_or_path}_{args.frac}_{args.method}_{args.dirichlet_alpha}_plugin_{args.plugin}_unused', test_loss, iter)
    
def debug_info_print(iter, test_acc, test_loss, best_acc, patience_counter, args):
    if args.debug:
        debug_info = (f"Round: {iter}, "
                      f"Test accuracy: {test_acc:.2f}, "
                      f"Test loss: {test_loss:.2f}, "
                      f"Best accuracy: {best_acc:.2f}, "
                      f"Patience: {patience_counter}")
        print(debug_info)

def local_learning(args, edge_datasets, net_glob, idx):
    local = EdgeOpt(args=args, train_model=copy.deepcopy(net_glob), edge_dataset=edge_datasets[idx])
    w = local.train(global_net=copy.deepcopy(net_glob))
    return w

def sequential_compute(args, edge_datasets, net_glob, idxs_users):
    w_locals = []
        
    for idx in idxs_users:
        w = local_learning(args, edge_datasets, copy.deepcopy(net_glob), idx)
        w_locals.append(w)
    
    return w_locals

def federated_learning(args, edge_datasets, dataset_test, net_glob):
    print(f'start federated learning with {args.model_name_or_path} and {args.method}!')
    w_glob = net_glob.state_dict()
    best_acc, patience_counter = 0, 0

    for iter in trange(args.global_ep):
        
        if args.plugin:
            paraphrased_file = f'./paraphrased/{args.syn}'
            balanced_edge_datasets = apply_iid_approximation(copy.deepcopy(edge_datasets), paraphrased_file)
            idxs_users = balanced_sampling(edge_datasets, int(args.num_edges * args.frac))
            w_locals = sequential_compute(args, balanced_edge_datasets, net_glob, idxs_users)
        else:
            idxs_users = np.random.choice(range(args.num_edges), max(int(args.frac * args.num_edges), 1), replace=True)
            w_locals = sequential_compute(args, edge_datasets, net_glob, idxs_users)
        
        w_glob = server_avg(w_locals)
        net_glob.load_state_dict(w_glob)
        
        test_acc, test_loss = test_model(copy.deepcopy(net_glob), dataset_test, args)
        best_acc, patience_counter, early_stop = early_stop_check(test_acc, best_acc, patience_counter, args)
        
        torch.hpu.synchronize()
                
        if args.tsboard:
            log_performance(iter, test_acc, test_loss, best_acc, args)
        
        if args.debug:
            debug_info_print(iter, test_acc, test_loss, best_acc, patience_counter, args)

        if early_stop:
            print('early stopped federated training!')
            break

if __name__ == '__main__':
    args = args_parser()
    args.device = torch.device('hpu')
    
    writer = SummaryWriter(f'runs/') if args.tsboard else None

    init_seed(args.seed)
    dataset_train, dataset_test = load_data()    
    edge_datasets = partition_data(dataset_train, args)
    net_glob = initialize_global_model(args)
    
    federated_learning(args, edge_datasets, dataset_test, net_glob)

    if args.tsboard:
        writer.close()