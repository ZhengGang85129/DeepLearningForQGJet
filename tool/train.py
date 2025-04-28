import torch
import torch.nn as nn 
from typing import Tuple
import time
import logging
import argparse
from torch.utils.data import DataLoader
import torch.optim.lr_scheduler as lr_scheduler
import shutil
import os, sys
sys.path.append(os.getcwd())
from JetTagger.utils.tool import set_seed, AverageMeter, Monitor
import JetTagger.utils.config as config
from JetTagger.model.model import *
from JetTagger.utils.read_model_configs import read_model_configs
from JetTagger.utils.Validation import _has_directory
from JetTagger.utils.dataset_configs import DATASET_CONFIGS
import yaml
import collections
from datetime import datetime
from torch import autocast, GradScaler

def get_device() -> torch.device:
    
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("CUDA is available. Using GPU.") 
    else:
        print("Using CPU.")
    return device

def get_logger() -> logging.Logger:
    logger_name = "JetTagger-Logger"
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = "[%(asctime)s %(levelname)s %(filename)s line %(lineno)d %(process)d %(message)s]"
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    now = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join('./', f"training-{now}.log") 
    return logger

def get_parser() -> argparse.ArgumentParser:
    parser  = argparse.ArgumentParser(
        description = "Jet Tagger" 
    )
    parser.add_argument('--config', type = str, default = None)
    args = parser.parse_args()
    assert args.config is not None
    
    cfg = config.load_cfg_from_cfg_file(args.config) #FIXME
    return cfg 

def create_dataloaders(args):
    if args.data_name not in DATASET_CONFIGS:
        raise ValueError(f'Dataset {args.data_name} not supported.')
    cfg = DATASET_CONFIGS[args.data_name]
    
    train_dataset = cfg['dataset_class'](
        split = cfg['train_split'],
        prepare_fn = cfg['prepare_fn'],
        batch_shuffle = True if args.data_name == 'PrivateDataset' else True,
        data_shuffle = True if args.data_name == 'PrivateDataset' else False,
        shuffle = True if args.data_name == 'JetClass_QuarkGluon' else False,
        datasets_root=args.data_root
    )
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = 1 if args.data_name == 'PrivateDataset' else args.batch_size_train,
        shuffle = False,
        drop_last = False,
        pin_memory = True,
        collate_fn=cfg['collate_fn'],
        num_workers= args.num_workers 
    )

    val_dataloader = None
    
    if args.evaluate:
        val_dataset = cfg['dataset_class'](
            split=cfg['val_split'],
            prepare_fn=cfg['prepare_fn'],
            batch_shuffle=True if args.data_name == 'PrivateDataset' else False,
            data_shuffle=True if args.data_name == 'PrivateDataset' else False,
            datasets_root=args.data_root
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=1 if args.data_name == 'PrivateDataset' else args.batch_size_validation,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=cfg['collate_fn'],
            num_workers=args.num_workers
        )
    
    return train_dataloader, val_dataloader

def create_dataloaders(args):
    if args.data_name not in DATASET_CONFIGS:
        raise ValueError(f"Dataset {args.data_name} not supported.")
    
    cfg = DATASET_CONFIGS[args.data_name]

    train_dataset = cfg['dataset_class'](
        split=cfg['train_split'],
        prepare_fn=cfg['prepare_fn'],
        batch_shuffle=cfg['batch_shuffle'],
        data_shuffle=cfg['data_shuffle'],
        datasets_root=args.data_root
    )
    
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=cfg['batch_size_train'],
        shuffle=False,
        drop_last=False,
        pin_memory=True,
        collate_fn=cfg['collate_fn'],
        num_workers=args.num_workers
    )
    
    val_dataloader = None
    if args.evaluate:
        val_dataset = cfg['dataset_class'](
            split=cfg['val_split'],
            prepare_fn=cfg['prepare_fn'],
            batch_shuffle=cfg['batch_shuffle'],
            data_shuffle=cfg['data_shuffle'],
            datasets_root=args.data_root
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=cfg['batch_size_val'],
            shuffle=False,
            drop_last=False,
            pin_memory=True,
            collate_fn=cfg['collate_fn'],
            num_workers=args.num_workers
        )
    
    return train_dataloader, val_dataloader

def main():
    global args, logger, device
    args = get_parser()
    logger = get_logger() 
    device = get_device() 
    if args.manual_seed is not None:
        set_seed(args.manual_seed)    
    
    train_dataloader, val_dataloader = create_dataloaders(args)
     
    
    with open(args.model_cfg_file, 'r') as file:
        
        model_parameters = yaml.load(file, Loader = yaml.FullLoader) 
    
    if args.model_name == 'MomentumCloudNet':
        model  = eval(args.model_name)(in_channels = args.in_channels, num_classes = args.classes, **read_model_configs(model = args.model_name, parameters = model_parameters))
    else:
        model = eval(args.model_name)(**read_model_configs(model = args.model_name, parameters = model_parameters))
    
    
    _has_directory(args.save_path)
    logger.info(f"=> create {args.save_path} if this directory does not exist")
     
    np = sum(p.numel() for p in model.parameters() if p.requires_grad)
     
    logger.info(args)
    logger.info("=> creating model")
    logger.info(f"Total number of parameters in model: {np}")
    logger.info(f"Classes: {args.classes}")
    logger.info(model)
    
    criterion = nn.CrossEntropyLoss(reduction = 'none')
    optimizer = torch.optim.RAdam(model.parameters(), 
                                  betas=(0.95, 0.999), 
                                  eps = args.eps, 
                                  lr = args.base_lr)
    
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 
        mode = 'min', 
        threshold_mode = 'abs', 
        threshold = 0.001, 
        cooldown = 0, 
        factor = 0.5, 
        patience = 1, 
        verbose = True)

    model.to(device)
    scaler = GradScaler()
    
    ############################
    # Resume the training state
    ############################
    if (args.resume is not None) and args.resume: 
        checkpoint = torch.load(args.ckpt) 
        state_dict = checkpoint["state_dict"]
        new_state_dict = collections.OrderedDict()
        optimizer_state = checkpoint['optimizer']
        if checkpoint.get('scheduler') is not None:
            scheduler_state = checkpoint['scheduler'] 
            scheduler.load_state_dict(scheduler_state)
        for k, v in state_dict.items():
            new_state_dict[k] = v
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(optimizer_state)
        logger.info(f"=> loaded ckpt : {args.ckpt} (epoch {checkpoint['epoch']})")

    logger.info(optimizer)
    logger.info(scheduler)
    min_val_loss = float('inf')
    accumulate_stop_steps = 0
    # monitor -> Help for plotting the loss curve and Accuracy during training on train/val dataset.
    monitor = Monitor(save_path = args.save_path, resume = args.resume) 
    state = {
        'train': {
            'mLoss': [],
            'mAcc': [],
        },
        'eval-train': {
            'mLoss': [],
            'mAcc': []
        },
        'val': {
            'mLoss': [],
            'mAcc': [],
        },
        'best_point': {
            'mLoss': float('inf'),
            'mAcc': -float('inf'),
            'current_epoch': 0
        }
    }
    
    epoch_offset = checkpoint['epoch'] if args.resume else 0 
    
    for epoch in range(args.start_epoch, args.epochs):
        epoch_log = epoch + 1 + epoch_offset
        
        mLoss_train, mAcc_train = train(train_loader = train_dataloader, 
                                        model = model, 
                                        criterion = criterion, 
                                        epoch = epoch_log - 1, 
                                        optimizer = optimizer, scaler = scaler)
        
        state['train']['mLoss'].append(mLoss_train)
        state['train']['mAcc'].append(mAcc_train)
    
        mAcc_evaltrain, mAcc_val, mLoss_evaltrain, mLoss_val =  validate(train_loader = train_dataloader, val_loader = val_dataloader, model = model, criterion= criterion, scaler = scaler)
        
        state['eval-train']['mAcc'].append(mAcc_evaltrain)
        state['eval-train']['mLoss'].append(mLoss_evaltrain)
        state['val']['mAcc'].append(mAcc_val)
        state['val']['mLoss'].append(mLoss_val)
        scheduler.step(mLoss_val)
         
        if ( - mLoss_val + min_val_loss) >= 0.00001: 
            state['best_point']['mLoss'] = mLoss_val 
            state['best_point']['mAcc'] = mAcc_val 
            state['best_point']['current_epoch'] = epoch_log
            min_val_loss = mLoss_val
            is_best = True
            accumulate_stop_steps = 0 
        else: 
            accumulate_stop_steps += 1 
            is_best = False
        if (epoch_log % args.save_freq == 0):
            filename = args.save_path + '/model_last.pt'
            logger.info('Saving checkpoing to: '+ filename)
            torch.save({
                'epoch': epoch_log,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'is_best': is_best}, filename)
        if is_best:
            logger.info('Best validation parameters updated to checkpoint')
            shutil.copy(filename, os.path.join(args.save_path, 'model_best.pt')) 
        else:
            logger.info(f'Accumulate steps: {accumulate_stop_steps}')
         
        
        monitor.update(state = state)
           
        if accumulate_stop_steps == args.early_stop:
            logger.info('Accumulate steps rechieved early stop steps')
            
            break     
        
        
        

def train(train_loader: torch.utils.data.DataLoader, model:nn.Module, criterion: nn.modules.loss._WeightedLoss, optimizer: torch.optim.Optimizer, epoch: int, scaler: torch.GradScaler) -> float:
    logger.info(">>>>>>>>>>>>>>>>>>>> Start Training <<<<<<<<<<<<<<<<<<<<<<<")

    model.train()
    
    batch_time = AverageMeter()
    data_time = AverageMeter()
    loss_meter = AverageMeter()
    correct_meter = AverageMeter()
    weight_meter = AverageMeter()
    
    max_iter = args.epochs * len(train_loader)
    dataset_per_epoch = len(train_loader)
    
    end = time.time()
    for b_index, data in enumerate(train_loader):
        data_time.update(time.time() - end)
            
        momentum = data.get('particle_momentums', None).to(device, non_blocking = True)
        feat = data.get('particle_features', None).to(device, non_blocking = True)
        target = data.get('y', None).to(device, non_blocking = True)
        
        with autocast(device_type=device.type):
            if args.model_name == 'ParticleTransformer':
                input = [feat, momentum]
                output = model(*input, return_attn = False)
            elif args.model_name == 'ParticleNet':
                input = torch.cat([momentum, feat], dim = -1)
                output = model(input)
            else:
                input = [feat, momentum]
                output = model(input, return_attn = False)
            
            loss = criterion(output, target)
        loss = loss.sum()
        weight_loss = loss.sum()
        optimizer.zero_grad()
        scaler.scale(weight_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_time.update(time.time() - end)
        end = time.time()
        
        current_iter = epoch * dataset_per_epoch + b_index + 1 
        remain_iter = max_iter - current_iter
        remain_time = batch_time.avg_value * remain_iter 
        
        t_m, t_s = divmod(remain_time, 60)
        t_h, t_m = divmod(t_m, 60)
        remain_time = '{:02d}:{:02d}:{:02d}'.format(int(t_h), int(t_m), int(t_s))
        loss_meter.update(loss.item())
        correct_meter.update(((output.argmax(1) == target).type(torch.float)).sum().item()) 
        
        if ((b_index + 1) % args.print_freq) == 0:
            
            logger.info(f'Epoch: [{epoch + 1}/{args.epochs}][{b_index + 1}/{dataset_per_epoch}] '
                        f'Data {data_time.current_value:.3f} s ({data_time.avg_value:.3f} s) '
                        f'Batch {batch_time.current_value:.3} s ({batch_time.avg_value:.3f} s) '
                        f'Remain(estimation) {remain_time} '
                        f'Loss {(loss_meter.current_value / weight_meter.current_value):.4f} ({(loss_meter.sum / weight_meter.sum):.4f}) '
                        f'Acc {(correct_meter.current_value / weight_meter.current_value):.4f} ({(correct_meter.sum / weight_meter.sum):.4f})'
                        )
            
    return loss_meter.sum/weight_meter.sum, correct_meter.sum/weight_meter.sum
         
def validate(train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, model:nn.Module, criterion: nn.modules.loss._WeightedLoss, scaler: GradScaler) -> Tuple[float]:
    
    logger.info(">>>>>>>>>>>>>>>>>>>> Start Evaluation <<<<<<<<<<<<<<<<<<<<<<<")
    
    batch_time = dict()
    data_time = dict()
    loss_meter = dict()
    correct_meter = dict()
    for dataset in ['val', 'train']:
        batch_time[dataset] = AverageMeter()
        data_time[dataset] = AverageMeter()
        loss_meter[dataset] = AverageMeter()
        correct_meter[dataset] = AverageMeter()
         
    model.eval()
    end = time.time()
     
    with torch.no_grad(): 
        if args.evaluate_train:
            for b_index, data in enumerate(train_loader):
                
                coord = data.get('particle_momentums', None).to(device, non_blocking = True)
                feat = data.get('particle_features', None).to(device, non_blocking = True)
                target = data.get('y', None).to(device, non_blocking = True)

                with autocast(device_type=device.type):
                    if args.model_name == 'ParticleTransformer':
                        input = [feat, coord]
                        output = model(*input, return_attn = False)
                    elif args.model_name == 'ParticleNet':
                        input = torch.cat([coord, feat], dim = -1)
                        output = model(input)
                    else:
                        input = [feat, coord]
                        output = model(input, return_attn = False)
                
                    loss = criterion(output, target)
                loss = loss.sum()
                
                batch_time['train'].update(time.time() - end) 
                loss_meter['train'].update(loss.item())
                correct_meter['train'].update(((output.argmax(1) == target).type(torch.float)).sum().item() )
                
                end = time.time() 
                if (b_index + 1) % args.print_freq == 0:
                    logger.info(f'(Train dataset) Batch: [{b_index + 1}/{len(train_loader)}] '
                                f"Data {data_time['train'].current_value:.3f} s ({data_time['train'].avg_value:.3f} s) "
                                f"Batch {batch_time['train'].current_value:.3f} s ({batch_time['train'].avg_value:.3f} s) "
                                f"Loss {(loss_meter['train'].current_value):.4f} ({(loss_meter['train'].sum):.4f}) "
                                f"Acc {(correct_meter['train'].current_value):.4f} ({(correct_meter['train'].sum ):.4f})"
                                )
        end = time.time() 
        for b_index, data in enumerate(val_loader):
            
            momentum = data.get('particle_momentums', None).to(device, non_blocking = True)
            feat = data.get('particle_features', None).to(device, non_blocking = True)
            target = data.get('y', None).to(device, non_blocking = True)

            if args.model_name == 'ParticleTransformer':
                input = [feat, momentum]
                output = model(*input)
            elif args.model_name == 'ParticleNet':
                input = torch.cat([momentum, feat], dim = -1)
                output = model(input)
            else:
                input = [feat, momentum]
                output = model(input)
            loss = criterion(output, target)
            loss = loss.sum()
            
            batch_time['val'].update(time.time() - end) 
            loss_meter['val'].update(loss.item())
            correct_meter['val'].update(((output.argmax(1) == target).type(torch.float)).sum().item() )
            
            end = time.time() 
            if (b_index + 1) % args.print_freq == 0:
                logger.info(f'(Valid dataset) Batch: [{b_index + 1}/{len(val_loader)}] '
                            f"Data {data_time['val'].current_value:.3f} s ({data_time['val'].avg_value:.3f} s) "
                            f"Batch {batch_time['val'].current_value:.3f} s ({batch_time['val'].avg_value:.3f} s) "
                            f"Loss {(loss_meter['val'].current_value ):.4f} ({(loss_meter['val'].sum):.4f}) "
                            f"Acc {(correct_meter['val'].current_value ):.4f} ({(correct_meter['val'].sum ):.4f})"
                            )
    
    if args.evaluate_train:
        mAcc_train = (correct_meter['train'].sum)
        mLoss_train = loss_meter['train'].sum 
    else:
        mAcc_train = mLoss_train = 0 
    mAcc_val = (correct_meter['val'].sum )
    mLoss_val = loss_meter['val'].sum 
    return mAcc_train, mAcc_val, mLoss_train, mLoss_val
     

 


if __name__ == '__main__':
    main()  
    






