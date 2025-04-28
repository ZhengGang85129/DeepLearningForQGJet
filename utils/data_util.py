import numpy as np
import random
import torch
from typing import Tuple, List, Union, Generator
import pyarrow.parquet as pq
import pyarrow as pa 
import awkward as ak


def collate_fn_PrivateDataset(batch_tensor:List[torch.Tensor]) -> Tuple[torch.Tensor]:
    batch_tensor = batch_tensor[0]
    return batch_tensor


def pad_and_trunc(a: Union[np.ndarray, ak.Array], maxlen: int, value = 0, dtype = 'float32') -> Union[np.ndarray, ak.Array]:
    
    FLAG = False
    if (isinstance(a, np.ndarray) or isinstance(a, ak.Array)):
        
        if hasattr(a, 'shape'):
            LEN = a.shape[0]
            if a.shape[0] == maxlen:
                FLAG = True
        else:
            LEN = len(a)
            if len(a) == maxlen:
                FLAG = True
        if FLAG:return a  
        else:
            x = (np.zeros((maxlen)).astype(dtype))
            maxlen = min(maxlen, LEN)
            x[:maxlen] = a[:maxlen]
            return x 

def collate_fn_OpenDataset(batch_tensor) -> Tuple[torch.Tensor]:
    

    result = {key: None for key in ['y', 'particle_momentums', 'particle_features', 'weight']}    
    
    n_batch, n_pts_perevent = len(batch_tensor), batch_tensor[0]['particle_momentum'].shape[0]
    
    
    
    for k in result.keys():
        result[k] = torch.stack([batch_tensor[idx][k] for idx in range(n_batch)])
        if k == 'weight' or k == 'y':
            result[k] = torch.squeeze(result[k], -1)
            continue
    
     
    return result
def collate_fn_OpenDataset_test(batch_tensor) -> Tuple[torch.Tensor]:
    

    result = {key: None for key in ['y', 'particle_momentums', 'particle_features', 'weight', 'indicator']}    
    
    n_batch, n_pts_perevent = len(batch_tensor), batch_tensor[0]['particle_momentums'].shape[0]
    
    
    
    for k in result.keys():
        result[k] = torch.stack([batch_tensor[idx][k] for idx in range(n_batch)])
        if k == 'weight' or k == 'y' or k == 'indicator':
            result[k] = torch.squeeze(result[k], -1)
            continue
    
     
    return result


 
def prepare_fn_OpenDataset(table: pa.Table, max_length = 50, pf_features: List[str] = None, pf_vectors: List[str] = None) -> Generator:
    '''
    Configure the input variables
    '''
     
    for (_, data) in table.to_pandas().iterrows():
    
        X = {key: None for key in pf_features}
        V = {key: data[key] for key in pf_vectors}
        Y = data['label']

        part_pt = np.hypot(data['part_px'], data['part_py'])  
        X['part_pt_log'] = np.log(part_pt)
        X['part_e_log'] = np.log(data['part_energy'])         
        X['part_ptrel'] = np.log(part_pt / data['jet_energy']) 
        X['part_erel'] = np.log(data['part_energy'] / data['jet_energy']) 
        X['part_deltaR'] = np.hypot(data['part_deta'], data['part_dphi']) 
        X['part_isCHad'] = (np.abs(data['part_pid']) == 211) + (np.abs(data['part_pid']) == 321) * 0.5 + (np.abs(data['part_pid']) == 2212) * 0.2
        X['part_isNHad'] = (np.abs(data['part_pid']) == 130) + (np.abs(data['part_pid']) == 2112) * 0.2
        
        X_stacks = []        
        for k, v in X.items():
            if v is None:
                X[k] = pad_and_trunc(data[k], maxlen = max_length)
            else:
                if ('isMuon' in k) or ('isPhoton' in k) or ('isElectron' in k):
                    X[k] = pad_and_trunc(v, maxlen = max_length, dtype = 'float32') 
                else:
                    X[k] = pad_and_trunc(v, maxlen = max_length)
            X_stacks.append(np.expand_dims(X[k], axis = 0)) 
        V_stacks = []        
        for k, v in V.items():
            V[k]  = pad_and_trunc(v, maxlen = max_length)
        #    X_stacks.append(np.expand_dims(V[k], axis = 0))
            V_stacks.append(np.expand_dims(V[k], axis = 0))
        X = torch.tensor(np.concatenate(X_stacks, axis = 0), dtype = torch.float32)
        V = torch.tensor(np.concatenate(V_stacks, axis = 0), dtype = torch.float32)
        
        del X_stacks
        del V_stacks 
        yield {'y': torch.tensor(Y, dtype = torch.int64).unsqueeze(0), 'particle_features': X.permute(1, 0), 'particle_momentums': V.permute(1, 0), 'weight': torch.ones((1))}
        
def prepare_fn_OpenDataset_test(table: pa.Table, max_length = 50, pf_features: List[str] = None, pf_vectors: List[str] = None) -> Generator:
    '''
    Configure the input variables
    '''
     
    for (_, data) in table.to_pandas().iterrows():
    
        X = {key: None for key in pf_features}
        V = {key: data[key] for key in pf_vectors}
        Y = data['label']
        Indicator = data['jet_pt']

        part_pt = np.hypot(data['part_px'], data['part_py'])  
        X['part_pt_log'] = np.log(part_pt)
        X['part_e_log'] = np.log(data['part_energy'])         
        X['part_ptrel'] = np.log(part_pt / data['jet_energy']) 
        X['part_erel'] = np.log(data['part_energy'] / data['jet_energy']) 
        X['part_deltaR'] = np.hypot(data['part_deta'], data['part_dphi']) 
        X['part_isCHad'] = (np.abs(data['part_pid']) == 211) + (np.abs(data['part_pid']) == 321) * 0.5 + (np.abs(data['part_pid']) == 2212) * 0.2
        X['part_isNHad'] = (np.abs(data['part_pid']) == 130) + (np.abs(data['part_pid']) == 2112) * 0.2
        
        X_stacks = []        
        for k, v in X.items():
            if v is None:
                X[k] = pad_and_trunc(data[k], maxlen = max_length)
            else:
                if ('isMuon' in k) or ('isPhoton' in k) or ('isElectron' in k):
                    X[k] = pad_and_trunc(v, maxlen = max_length, dtype = 'float32') 
                else:
                    X[k] = pad_and_trunc(v, maxlen = max_length)
            X_stacks.append(np.expand_dims(X[k], axis = 0)) 
        V_stacks = []        
        for k, v in V.items():
            V[k]  = pad_and_trunc(v, maxlen = max_length)
            V_stacks.append(np.expand_dims(V[k], axis = 0))
        #    X_stacks.append(np.expand_dims(V[k], axis = 0))
        X = torch.tensor(np.concatenate(X_stacks, axis = 0), dtype = torch.float32)
        V = torch.tensor(np.concatenate(V_stacks, axis = 0), dtype = torch.float32)
        
        del X_stacks
        del V_stacks 
        yield {'y': torch.tensor(Y, dtype = torch.int64).unsqueeze(0), 'particle_features': X.permute(1, 0), 'particle_momentums': V.permute(1, 0), 'weight': torch.ones((1)), 'indicator': torch.tensor(Indicator, dtype = torch.float64)}
        
         
def prepare_fn_PrivateDataset(file_label: List[Tuple[str, int]], keys: List[str] = [], shuffle: bool = True) -> Tuple[torch.Tensor]:
    '''
    Here, we need to concatenate two different arrays of flavour.
    This function prepare one batch per iteration.
    ''' 
    data = dict()
    Y = list()
    for file, label in file_label:
        with np.load(file) as X:
            for key in keys:
                if data.get(key, None) is None:
                    data[key] = X[key] if key != 'jet' else X[key][:, 0]
                else:
                    data[key] = np.concatenate((data[key], X[key])) if key != 'jet' else np.concatenate((data[key], X[key][:, 0]))
            Y.extend([label] * len(X[keys[0]]))
    ##Data preprocess ##   
    ###<<< FIXME >>>###          
    for key in keys:
        if key == 'pf' or key == 'hgcal':
            if key == 'pf':
                with np.errstate(divide='ignore', invalid='ignore'):
                    data[key][:, :, 11:13] = np.log(data[key][:, :, 11:13])
            else:
                with np.errstate(divide='ignore', invalid='ignore'):
                    data[key] = np.log(data[key]) 
            data[key][np.isneginf(data[key]) | np.isnan(data[key])] = 0
        
        data[key] = torch.nan_to_num(torch.tensor(data[key], dtype = torch.float32)) 
    data['y'] = torch.tensor(Y, dtype = torch.int64)
    if shuffle:
        shuffle_index = torch.randperm(len(data['y']))
        for k, v in data.items():
            data[k] = v[shuffle_index]
    
    return data
    
    