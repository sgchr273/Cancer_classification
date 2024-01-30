
#  python  baselines.py --DEBUG 8000 
import pickle
from torchvision import transforms
import argparse
from dataset import get_dataset, get_handler
from model import CancerModel
import os
import numpy as np
import torch
import torch.nn as nn
import time


from saving import load_model, save_queried_idx
from run import exper, opts, decrease_dataset
if __name__=="__main__":

    DATA_NAME = opts.data
    SAVE_FILE = opts.savefile
    NUM_QUERY = opts.nQuery

    # non-openml data defaults
    args_pool = {'MNIST':
                    {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                    'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
                'FashionMNIST':
                    {'n_epoch': 10, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]),
                    'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.01, 'momentum': 0.5},
                    'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, ), (0.2470, ))])},
                'SVHN':
                    {'n_epoch': 20, 'transform': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970))]),
                    'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 1000, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.01, 'momentum': 0.5}},
                'CIFAR10':
                    {'n_epoch': 3, 'transform': transforms.Compose([ 
                        transforms.RandomCrop(32, padding=4),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))
                    ]),
                    'loader_tr_args':{'batch_size': 128, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 100, 'num_workers': 1}, # change back to 1000
                    'optimizer_args':{'lr': 0.05, 'momentum': 0.3},
                    'transformTest': transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])},
                'BreaKHis':
                    {'n_epoch': 20, 'transform':transforms.Compose([
                        transforms.RandomRotation(20),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=(0.2, 0.2)),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet mean and std
                    ]),
                    'loader_tr_args':{'batch_size': 64, 'num_workers': 1},
                    'loader_te_args':{'batch_size': 64, 'num_workers': 1},
                    'optimizer_args':{'lr': 0.000001, 'momentum': 0.5},
                    'transformTest' : transforms.Compose([
                        transforms.RandomRotation(20),
                        transforms.RandomHorizontalFlip(),
                        transforms.RandomVerticalFlip(),
                        transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.1),
                        transforms.RandomAffine(degrees=0, translate=(0.2, 0.2), shear=(0.2, 0.2)),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
                    }}
    
    args = args_pool[DATA_NAME]
    args['lr'] = opts.lr
    args['modelType'] = opts.model
    args['fishIdentity'] = opts.fishIdentity
    args['fishInit'] = opts.fishInit
    args['lamb'] = opts.lamb
    args['backwardSteps'] = opts.backwardSteps
    args['pct_top'] = opts.pct_top
    args['chunkSize'] = opts.chunkSize

    args['savefile'] = SAVE_FILE


if __name__=="__main__":

    X_tr, X_te, Y_tr, Y_te = get_dataset()
    if opts.DEBUG:
        X_tr, Y_tr = decrease_dataset(X_tr, Y_tr)
    opts.dim = np.shape(X_tr)[1:]
    handler = get_handler('BreaKHis')

    n_pool = len(Y_tr)
    n_test = len(Y_te)
    NUM_INIT_LB = opts.nStart
    idxs_lb = np.zeros(n_pool, dtype=bool)
    idxs_tmp = np.arange(n_pool)
    np.random.shuffle(idxs_tmp)
    idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True
    # print('initial pool:', sum(idxs_lb))
    print('number of labeled pool: {}'.format(NUM_INIT_LB), flush=True)
    print('number of unlabeled pool: {}'.format(n_pool - NUM_INIT_LB), flush=True)
    print('number of testing pool: {}'.format(n_test), flush=True)

    net = CancerModel(num_classes=2)
    # for i in range(1):
    #     opts.savefile = SAVE_FILE + str(i)
    #     net = CancerModel(num_classes=2)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = np.zeros(n_pool, dtype=bool)
    #     idxs_tmp = np.arange(n_pool)
    #     np.random.shuffle(idxs_tmp)
    #     idxs_lb[idxs_tmp[:NUM_INIT_LB]] = True
    #     init_labeled = np.copy(idxs_lb)
    #     with open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + '.p', "wb") as savefile:
    #         pickle.dump(init_labeled, savefile)
    #     exper('entropy', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME)        
    # net = CancerModel(num_classes=2)
    # for i in range(1):
    #     opts.savefile = SAVE_FILE + str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
    #     exper('bait', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME) 

        
    # net = CancerModel(num_classes=2)
    # for i in range(1):
    #     opts.savefile = SAVE_FILE + str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
    #     exper('rand', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME) 
    
    # net = CancerModel(num_classes=2)
    # for i in range(1):
    #     opts.savefile = SAVE_FILE + str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
    #     exper('margin', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME) 
    
    # net = CancerModel(num_classes=2)
    # for i in range(1):
    #     opts.savefile = SAVE_FILE + str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
    #     exper('badge', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME) 

    # net = CancerModel(num_classes=2)
    # for i in range(1):
    #     opts.savefile = SAVE_FILE + str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
    #     exper('kmeans', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME) 

    # net = CancerModel(num_classes=2)
    # for i in range(1):
    #     opts.savefile = SAVE_FILE + str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
    #     exper('lcs', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME) 

    net = CancerModel(num_classes=2)
    for i in range(5):
        opts.savefile = SAVE_FILE + str(i)
        # load_model(1, net, opts.savefile, 'entropy')
        # idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
        # opts.savefile = SAVE_FILE + str(i) + "_standard"
        exper('fish', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME)

    # net = CancerModel(num_classes=2)
    # for i in range(5):
    #     opts.savefile = SAVE_FILE + str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
    #     opts.savefile = SAVE_FILE + str(i) + "_dispersed"
    #     exper('fish', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME, method="dispersed")

    # net = cancer_model(num_classes=2)
    # for i in range(10):
    #     opts.savefile = SAVE_FILE + str(i)
    #     load_model(1, net, opts.savefile, 'entropy')
    #     idxs_lb = pickle.load(open("./Save/Queried_idxs/InitLabeled_" + opts.savefile + ".p", "rb"))
    #     opts.savefile = SAVE_FILE + str(i) + "_relative"
    #     exper('FISH', X_tr, Y_tr, idxs_lb, net, handler, args, X_te, Y_te, DATA_NAME, method="relative")
    
