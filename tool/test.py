import os, sys
sys.path.append(os.getcwd())
import time
import random
import matplotlib.pyplot as plt
import numpy as np
import logging
import argparse
import yaml
from typing import List 
import collections

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader

from JetTagger.utils import config
from JetTagger.utils.read_model_configs import read_model_configs
from JetTagger.model.model import *
from JetTagger.utils.tool import set_seed, AverageMeter
import JetTagger.utils.config as config

from JetTagger.utils.Validation import make_ROC_curve, make_probability

from JetTagger.utils.dataset_configs import TEST_DATASET_CONFIGS



def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
def get_parser() -> argparse.ArgumentParser:
    
    parser  = argparse.ArgumentParser(
        description = "Jet Tagger" 
    )
    parser.add_argument('--config', type = str, default = None)
    parser.add_argument('--ckpt', type = str, default = None)
    args = parser.parse_args()
    assert args.config is not None
    
    cfg = config.load_cfg_from_cfg_file(args.config) 
    return cfg 
    


def get_logger() -> logging.Logger:
    logger_name = "JetTagger-Logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d %(message)s]"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    
    return logger

def main() -> None:
    global args, logger, device
    args = get_parser()
    logger = get_logger()
    device = get_device()
    
    if args.manual_seed is not None:
        set_seed(args.manual_seed)    
    
    logger.info(args)
    
    logger.info("=> Creating Model ...")
    logger.info(f"=> Number of Classes: {args.classes}")

    with open(args.model_cfg_file, 'r') as file:
        model_parameters = yaml.load(file, Loader = yaml.FullLoader) 
    
    if args.model_name == 'MomentumCloudNet':
        model  = eval(args.model_name)(in_channels = args.in_channels, num_classes = args.classes, **read_model_configs(model = args.model_name, parameters = model_parameters))
    else:
        model = eval(args.model_name)(**read_model_configs(model = args.model_name, parameters = model_parameters))
    
    model = model.cuda()
    
    if os.path.isfile(args.ckpt):
        criterion = nn.CrossEntropyLoss(reduction = 'none')
        logger.info("=> Loading checkpoing")
        checkpoint = torch.load(args.ckpt) 
        state_dict = checkpoint["state_dict"]
        new_state_dict = collections.OrderedDict()
        
        for k, v in state_dict.items():
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict, strict = True)
        logger.info(f"=> loaded ckpt : {args.ckpt} (epoch {checkpoint['epoch']})")
        args.epoch = checkpoint['epoch']
        
        dataloader = create_dataloader()
        
        test(model, dataloader, criterion)
    else:
        raise RuntimeError(f"=> no checkpoint found at {args.ckpt}" )

def create_dataloader() -> torch.utils.data.DataLoader:
    
    if args.data_name not in TEST_DATASET_CONFIGS:
        raise ValueError(f'Dataset: {args.data_name} not supported.')
    cfg = TEST_DATASET_CONFIGS[args.data_name]
     
    dataset =  cfg['dataset_class'](
        split = 'test',        
        prepare_fn = cfg['prepare_fn'], 
        shuffle = True,
        datasets_root = args.data_root)
        
    dataloader = DataLoader(
        dataset, 
        batch_size = args.batch_size_test, 
        drop_last = False,
        collate_fn =  cfg['collate_fn'], 
        num_workers = args.num_workers)
    
    return dataloader

@torch.no_grad()
def test(model: nn.Module, data: torch.utils.data.DataLoader, criterion: nn.modules.loss._WeightedLoss) -> None: 
         
    logger.info(">>>>>>>>>>>>>> Start Testing <<<<<<<<<<<<<<")
    batch_time = AverageMeter()
    loss_meter = AverageMeter() 
    correct_meter = AverageMeter()
    end = time.time() 
    dataset_per_epoch = len(data)  
    
    Indicator = []
    pred = []
    true = []
    weight_list = []
    
    model.eval()
    
    with torch.no_grad():
        
        for b_index, data in enumerate(data):
            
                momentum = data.get('particle_momentums', None).to(device, non_blocking = True)
                feat = data.get('particle_features', None).to(device, non_blocking = True)
                target = data.get('y', None).to(device, non_blocking = True)
                
                indicator = data.get('indicator', None).to(device, non_blocking = True)

                if args.model_name == 'ParticleTransformer':
                    input = [feat, momentum]
                    output = model(*input)

                elif args.model_name == 'ParticleNet':
                    input = torch.cat([momentum, feat], dim = -1)
                    output = model(input)
                else:
                    input = [feat, momentum]
                    output = model(input)
                loss = criterion(output, target).sum() 
                
             
                
                batch_time.update(time.time() - end)
                
                loss_meter.update(loss.item())
                correct_meter.update(((output.argmax(1) == target).type(torch.float)).sum().item()) 
                end = time.time()
                
                
                Indicator.append(indicator)
                pred.append(output[:, 1])
                true.append(target)
                 
                if ((b_index + 1) % args.print_freq) == 0:
                    logger.info(
                        f'[{b_index + 1}/{dataset_per_epoch}] '
                        f'Batch {batch_time.current_value*(1e3):.3f} ms ({batch_time.avg_value*(1e3):.3f} ms) '
                        f'Loss {(loss_meter.current_value ):.4f} ({(loss_meter.sum ):.4f}) '
                        f'Acc {(correct_meter.current_value ):.4f} ({(correct_meter.sum):.4f})'
                        ) 
            
    
    logger.info(
        f'[{b_index + 1}/{dataset_per_epoch}] '
        f'Batch {batch_time.current_value*(1e3):.3f} ms ({batch_time.avg_value*(1e3):.3f} ms) '
        f'Loss {(loss_meter.current_value):.4f} ({(loss_meter.sum):.4f}) '
        f'Acc {(correct_meter.current_value):.4f} ({(correct_meter.sum):.4f})'
        ) 
    
    
    logger.info(f'=> Plot ROC curve, saved in folder: {args.save_path}')
    plot_roc_curce(Indicator, pred, true, weight_list) 

def plot_roc_curce(indicator: List, pred: List, true: List, weight: List) -> None:
    split_value = [500, 510, 520, 530, 550]
    '''
    Derived from the function: performance_reveal from utils.Validation
    '''
    indicator = torch.cat(indicator, dim = 0).to(torch.device('cpu')).numpy()
    pred = torch.cat(pred, dim = 0).to(torch.device('cpu')).numpy()
    true = torch.cat(true, dim = 0).to(torch.device('cpu')).numpy()
    weight = torch.cat(weight, dim = 0).to(torch.device('cpu')).numpy()
    
    scale = np.sum(weight[true>0.5]) / np.sum(weight[true<0.5])
    weight_copy = np.copy(weight)
    weight_copy[true<0.5] *= scale
    
    
     
    overall_auc = make_ROC_curve(  { f'overall' : (true, pred, weight) }, 'roc_curve_overall', args.save_path )

    # Extra check probability
    prob_dict = { 'Gluon' : (pred[ true < 0.5 ], weight_copy[ true < 0.5 ]), 'Quark' : (pred[ true > 0.5 ], weight_copy[ true > 0.5 ]) }
    make_probability (prob_dict, 'overall_hist',  args.save_path)

    # ROC curve with indicator splitting
    roc_auc = []
    for i in range(len(split_value)-1):

        cut = ( indicator > split_value[i] ) & ( indicator < split_value[i+1] )
        wgt = weight[cut]
        y   = true[cut]
        y_p = pred[cut]

        scale = np.sum(wgt[y>0.5]) / np.sum(wgt[y<0.5])
        wgt[y<0.5] *= scale

        roc_auc.append( make_ROC_curve(  { f'jet_pt_{split_value[i]}_{split_value[i+1]}' : (y, y_p, wgt) }, f'roc_curve_jet_pt_{split_value[i]}_{split_value[i+1]}', args.save_path ) )

        prob_dict = { 'Gluon' : (y_p[ y < 0.5 ], wgt[ y < 0.5 ]), 'Quark' : (y_p[ y > 0.5 ], wgt[ y > 0.5 ]) }
        make_probability (prob_dict, f'hist_jet_pt_{split_value[i]}_{split_value[i+1]}', args.save_path)


    # AUC value as a function of the given indicator
    split_value = np.array(split_value)
    var   = (split_value[:-1] + split_value[1:]) * 0.5
    error = (split_value[1:] - split_value[:-1]) * 0.5

    fig, ax = plt.subplots()
    ax.errorbar(var, np.array(roc_auc), xerr=error, fmt='o', label=f"Overall AUC = {overall_auc:>3f}")

    ax.legend()
    ax.set_xlabel('jet_pt [GeV]')
    ax.set_ylabel('AUC')
    plt.savefig(f'{args.save_path}/auc_jet_pt_split.png')
    plt.savefig(f'{args.save_path}/auc_jet_pt_split.pdf')
    np.savez(f"{args.save_path}/auc_jet_pt_split.npz", var=var, error=error, overall_auc=np.array(overall_auc), auc=np.array(roc_auc))
    plt.close(fig)
    
    
    return 

        
if __name__ == "__main__":
    main()