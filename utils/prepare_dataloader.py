import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from JetTagger.utils.customizedDataLoader import ModifiedDataLoader
import os

def prepare_combined_dataloader(dataset_dict : list, device):

    var_set = []
    X_set = []
    X_sup_set = []
    X_weight_set = []
    y_set = []

    filelist = []
    for path, target in dataset_dict.items():
        filelist += [ (path + '/' + ifile, target) for ifile in os.listdir(path) ]

    for f, target in filelist:

        var       = np.concatenate([ np.load(f)[ 'jet' ][:,0] ])
        X         = np.concatenate([ np.load(f)[ 'pf'  ]      ])
        X_sup     = np.concatenate([ np.load(f)[ 'hgcal' ]    ])
        X_weight  = np.concatenate([ np.load(f)[ 'wgt' ]      ])

        X[:,:,11:13] = np.log(X[:,:,11:13])
        X[np.isneginf(X) | np.isnan(X)]=0

        X_sup = np.log(X_sup)
        X_sup[np.isneginf(X_sup) | np.isnan(X_sup)]=0

        var      = torch.nan_to_num( torch.tensor(var, dtype=torch.float32 ) )
        X        = torch.nan_to_num( torch.tensor(X, dtype=torch.float32 ) )
        X_sup    = torch.nan_to_num( torch.tensor(X_sup, dtype=torch.float32 ) )
        X_weight = torch.nan_to_num( torch.tensor(X_weight, dtype=torch.float32 ) )
        y = torch.full((len(X),), target, dtype=torch.int64)

        var_set. append(var)
        X_set. append(X)
        X_sup_set. append(X_sup)
        X_weight_set.append(X_weight)
        y_set.append(y)

    var_set = torch.cat(var_set)
    X_set = torch.cat(X_set)
    X_sup_set = torch.cat(X_sup_set)
    X_weight_set = torch.cat(X_weight_set)
    y_set = torch.cat(y_set)

    # Weight balance
    scale = torch.sum(X_weight_set[y_set>0.5]) / torch.sum(X_weight_set[y_set<0.5])
    X_weight_set[y_set<0.5] *= scale

    ds = TensorDataset(var_set, X_set, X_sup_set, y_set, X_weight_set)
    ds_loader = DataLoader(ds, batch_size=1024, shuffle=True, num_workers=4, pin_memory=True)
    ds_loader = ModifiedDataLoader(ds_loader, device)

    return ds_loader


def prepare_dataloader(files : list, device, split=(0, 1)):

    '''
    Prepare data loader with input as python list format
    Input : [ ('signal_file.npz', 1), ('bkg_file.npz', 0) ]
    Output : DataLoader
    '''

    X_set = []
    X_weight_set = []
    y_set = []

    i_split, nsplit = split
    batch_size = int(512 / nsplit)

    for filename, target in files:

        X        = np.load(filename)['pf'][batch_size*i_split:batch_size*(i_split+1),:,:]
        X_weight = np.load(filename)['wgt'][batch_size*i_split:batch_size*(i_split+1)]

        # Temp
        #X = np.log(X)
        X[:,:,11:13] = np.log(X[:,:,11:13])
        X[np.isneginf(X) | np.isnan(X)]=0

        X        = torch.nan_to_num( torch.tensor(X, dtype=torch.float32 ) )
        X_weight = torch.nan_to_num( torch.tensor(X_weight, dtype=torch.float32 ) )
        y = torch.full((len(X),), target, dtype=torch.int64)

        X_set. append(X)
        X_weight_set.append(X_weight)
        y_set.append(y)

    X_set = torch.cat(X_set)
    y_set = torch.cat(y_set)
    X_weight_set = torch.cat(X_weight_set)

    ds = TensorDataset(X_set, y_set, X_weight_set)
    ds_loader = DataLoader(ds, batch_size=len(ds), shuffle=True, num_workers=4, pin_memory=True)
    ds_loader = ModifiedDataLoader(ds_loader, device)

    return next(iter(ds_loader))

def prepare_dataloader_performance(files : list, device, split=(0, 1)):

    '''
    Prepare data loader with input as python list format
    Input : [ ('signal_file.npz', 1), ('bkg_file.npz', 0) ]
    Output : DataLoader
    '''

    var_set = []
    X_set = []
    X_weight_set = []
    y_set = []

    i_split, nsplit = split
    batch_size = int(512 / nsplit)

    for filename, target in files:

        var      = np.load(filename)['jet'][batch_size*i_split:batch_size*(i_split+1)]
        X        = np.load(filename)['pf'][batch_size*i_split:batch_size*(i_split+1),:,:]
        X_weight = np.load(filename)['wgt'][batch_size*i_split:batch_size*(i_split+1)]

        # Temp
        #X = np.log(X)
        X[:,:,11:13] = np.log(X[:,:,11:13])
        X[np.isneginf(X) | np.isnan(X)]=0

        var      = torch.nan_to_num( torch.tensor(var, dtype=torch.float32 ) )
        X        = torch.nan_to_num( torch.tensor(X, dtype=torch.float32 ) )
        X_weight = torch.nan_to_num( torch.tensor(X_weight, dtype=torch.float32 ) )
        y = torch.full((len(X),), target, dtype=torch.int64)

        var_set. append(var)
        X_set. append(X)
        X_weight_set.append(X_weight)
        y_set.append(y)

    var_set = torch.cat(var_set)
    X_set = torch.cat(X_set)
    y_set = torch.cat(y_set)
    X_weight_set = torch.cat(X_weight_set)

    ds = TensorDataset(var_set, X_set, y_set, X_weight_set)
    ds_loader = DataLoader(ds, batch_size=len(ds), shuffle=True, num_workers=4, pin_memory=True)
    ds_loader = ModifiedDataLoader(ds_loader, device)

    return next(iter(ds_loader))

def prepare_dataloader_sup(files : list, device, split=(0, 1)):

    '''
    Prepare data loader with input as python list format
    Input : [ ('signal_file.npz', 1), ('bkg_file.npz', 0) ]
    Output : DataLoader
    '''

    X_set = []
    X_sup_set = []
    X_weight_set = []
    y_set = []

    i_split, nsplit = split
    batch_size = int(512 / nsplit)

    for filename, target in files:

        X        = np.load(filename)['pf'][batch_size*i_split:batch_size*(i_split+1),:,:]
        X_sup    = np.load(filename)['hgcal'][batch_size*i_split:batch_size*(i_split+1),:,:]
        X_weight = np.load(filename)['wgt'][batch_size*i_split:batch_size*(i_split+1)]

        # Temp
        X[:,:,11:13] = np.log(X[:,:,11:13])
        X[np.isneginf(X) | np.isnan(X)]=0

        X_sup = np.log(X_sup)
        X_sup[np.isneginf(X_sup) | np.isnan(X_sup)]=0

        X        = torch.nan_to_num( torch.tensor(X, dtype=torch.float32 ) )
        X_sup    = torch.nan_to_num( torch.tensor(X_sup, dtype=torch.float32 ) )
        X_weight = torch.nan_to_num( torch.tensor(X_weight, dtype=torch.float32 ) )
        y = torch.full((len(X),), target, dtype=torch.int64)

        X_set. append(X)
        X_sup_set. append(X_sup)
        X_weight_set.append(X_weight)
        y_set.append(y)

    X_set = torch.cat(X_set)
    X_sup_set = torch.cat(X_sup_set)
    y_set = torch.cat(y_set)
    X_weight_set = torch.cat(X_weight_set)

    ds = TensorDataset(X_set, X_sup_set, y_set, X_weight_set)
    ds_loader = DataLoader(ds, batch_size=len(ds), shuffle=True, num_workers=4, pin_memory=True)
    ds_loader = ModifiedDataLoader(ds_loader, device)

    return next(iter(ds_loader))


def prepare_dataloader_sup_performance(files : list, device, split=(0, 1)):

    '''
    Prepare data loader with input as python list format
    Input : [ ('signal_file.npz', 1), ('bkg_file.npz', 0) ]
    Output : DataLoader
    '''

    var_set = []
    X_set = []
    X_sup_set = []
    X_weight_set = []
    y_set = []

    i_split, nsplit = split
    batch_size = int(512 / nsplit)

    for filename, target in files:

        var      = np.load(filename)['jet'][batch_size*i_split:batch_size*(i_split+1),0]
        X        = np.load(filename)['pf'][batch_size*i_split:batch_size*(i_split+1),:,:]
        X_sup    = np.load(filename)['hgcal'][batch_size*i_split:batch_size*(i_split+1),:,:]
        X_weight = np.load(filename)['wgt'][batch_size*i_split:batch_size*(i_split+1)]

        # Temp
        X[:,:,11:13] = np.log(X[:,:,11:13])
        X[np.isneginf(X) | np.isnan(X)]=0

        X_sup = np.log(X_sup)
        X_sup[np.isneginf(X_sup) | np.isnan(X_sup)]=0

        var      = torch.nan_to_num( torch.tensor(var, dtype=torch.float32 ) )
        X        = torch.nan_to_num( torch.tensor(X, dtype=torch.float32 ) )
        X_sup    = torch.nan_to_num( torch.tensor(X_sup, dtype=torch.float32 ) )
        X_weight = torch.nan_to_num( torch.tensor(X_weight, dtype=torch.float32 ) )
        y = torch.full((len(X),), target, dtype=torch.int64)

        var_set. append(var)
        X_set. append(X)
        X_sup_set. append(X_sup)
        X_weight_set.append(X_weight)
        y_set.append(y)

    var_set = torch.cat(var_set)
    X_set = torch.cat(X_set)
    X_sup_set = torch.cat(X_sup_set)
    y_set = torch.cat(y_set)
    X_weight_set = torch.cat(X_weight_set)

    ds = TensorDataset(var_set, X_set, X_sup_set, y_set, X_weight_set)
    ds_loader = DataLoader(ds, batch_size=len(ds), shuffle=True, num_workers=4, pin_memory=True)
    ds_loader = ModifiedDataLoader(ds_loader, device)

    return next(iter(ds_loader))




def prepare_dataloader_ParT(files : list, device, split=(0, 1)):

    '''
    Prepare data loader with input as python list format
    Input : [ ('signal_file.npz', 1), ('bkg_file.npz', 0) ]
    Output : DataLoader
    '''

    X_set = []
    V_set = []
    X_weight_set = []
    y_set = []

    i_split, nsplit = split
    batch_size = int(512 / nsplit)

    for filename, target in files:

        X        = np.load(filename)['pf'][batch_size*i_split:batch_size*(i_split+1),:,:]
        V        = np.load(filename)['pf_var'][batch_size*i_split:batch_size*(i_split+1),:,:]
        X_weight = np.load(filename)['wgt'][batch_size*i_split:batch_size*(i_split+1)]

        # Temp
        X[:,:,11:13] = np.log(X[:,:,11:13])
        X[np.isneginf(X) | np.isnan(X)]=0
        V[np.isneginf(V) | np.isnan(V)]=0


        X        = torch.nan_to_num( torch.tensor(X, dtype=torch.float32 ) )
        V        = torch.nan_to_num( torch.tensor(V, dtype=torch.float32 ) )
        X_weight = torch.nan_to_num( torch.tensor(X_weight, dtype=torch.float32 ) )
        y = torch.full((len(X),), target, dtype=torch.int64)

        X_set. append(X)
        V_set. append(V)
        X_weight_set.append(X_weight)
        y_set.append(y)

    X_set = torch.cat(X_set)
    V_set = torch.cat(V_set)
    y_set = torch.cat(y_set)
    X_weight_set = torch.cat(X_weight_set)

    ds = TensorDataset(X_set, V_set, y_set, X_weight_set)
    ds_loader = DataLoader(ds, batch_size=len(ds), shuffle=True, num_workers=4, pin_memory=True)
    ds_loader = ModifiedDataLoader(ds_loader, device)

    return next(iter(ds_loader))


def prepare_dataloader_ParT_performance(files : list, device, split=(0, 1)):

    '''
    Prepare data loader with input as python list format
    Input : [ ('signal_file.npz', 1), ('bkg_file.npz', 0) ]
    Output : DataLoader
    '''

    var_set = []
    X_set = []
    V_set = []
    X_weight_set = []
    y_set = []

    i_split, nsplit = split
    batch_size = int(512 / nsplit)

    for filename, target in files:

        var      = np.load(filename)['jet'][batch_size*i_split:batch_size*(i_split+1),0]
        X        = np.load(filename)['pf'][batch_size*i_split:batch_size*(i_split+1),:,:]
        V        = np.load(filename)['pf_var'][batch_size*i_split:batch_size*(i_split+1),:,:]
        X_weight = np.load(filename)['wgt'][batch_size*i_split:batch_size*(i_split+1)]

        # Temp
        X[:,:,11:13] = np.log(X[:,:,11:13])
        X[np.isneginf(X) | np.isnan(X)]=0
        
        V[np.isneginf(V) | np.isnan(V)]=0


        var      = torch.nan_to_num( torch.tensor(var, dtype=torch.float32 ) )
        X        = torch.nan_to_num( torch.tensor(X, dtype=torch.float32 ) )
        V        = torch.nan_to_num( torch.tensor(V, dtype=torch.float32 ) )
        X_weight = torch.nan_to_num( torch.tensor(X_weight, dtype=torch.float32 ) )
        y = torch.full((len(X),), target, dtype=torch.int64)
        var_set. append(var)
        X_set. append(X)
        V_set. append(V)
        X_weight_set.append(X_weight)
        y_set.append(y)

    var_set = torch.cat(var_set)
    X_set = torch.cat(X_set)
    V_set = torch.cat(V_set)
    y_set = torch.cat(y_set)
    X_weight_set = torch.cat(X_weight_set)

    ds = TensorDataset(var_set, X_set, V_set, y_set, X_weight_set)
    ds_loader = DataLoader(ds, batch_size=len(ds), shuffle=True, num_workers=4, pin_memory=True)
    ds_loader = ModifiedDataLoader(ds_loader, device)

    return next(iter(ds_loader))
