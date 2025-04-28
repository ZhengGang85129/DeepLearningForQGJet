import os 
import sys
import random
import torch
import math
from torch.utils.data import Dataset, IterableDataset
from typing import Any, Callable,  List, Iterator, Tuple, Dict, Optional 
import pyarrow.parquet as pq
import pyarrow as pa
from itertools import cycle, chain, islice
import numpy as np
sys.path.append(os.getcwd())
sys.path.append(__file__)

from JetTagger.utils.tool import set_seed
from functools import partial


class QGJetDataset_v9pt15(Dataset):
    npz_keys = ['pf', 'pf_var', 'wgt'] #FIXME: Move to Config file in the near future
    Class = {'gluon': 0, 'quark': 1} #FIXME: Move to Config file in the near future
    batch_size = 512 #FIXME: Move to Config file in the near future
    datasets_root = '/home/public/hgcal_test_v9_pt15' #FIXME: Move to Config file in the near future
    '''
    Original dataset is already batched with batch size 512. 
    No need to specify batch size in dataloader.
    '''
    def __init__(self,
                split:str = 'training',
                batch_shuffle: bool = True,
                data_shuffle: bool = True,
                prepare_fn: Callable[[Any], Iterator] = None,
                datasets_root: str = '/home/public/hgcal_test_v9_pt15' 
                )-> None:

        super().__init__()
        
        self.split = split
        self.nClass = len(self.Class.keys())  
        self.batch_size = self.nClass * self.batch_size
        self.filelabel_list = self.__get_filelabel_list(datasets_root = self.datasets_root, shuffle = batch_shuffle) 
        self.prepare_fn = prepare_fn 
        self.datasets_root = datasets_root
        self.data_shuffle = data_shuffle
        assert os.path.isdir(self.datasets_root)
        
                 
    def __get_filelabel_list(self, datasets_root: str, shuffle: bool = True) -> List:
        

        #for flavour in self.Class.keys():

        datasets = dict() 
        
        for flavour in self.Class.keys():
            d_path = os.path.join(datasets_root, flavour)
            d_path = os.path.join(d_path, self.split)
            datasets[flavour] = [(os.path.join(d_path, item), self.Class[flavour]) for item in sorted(os.listdir(d_path))]
            if shuffle:
                random.shuffle(datasets[flavour])
        self.nbatch = len(datasets[list(self.Class.keys())[0]])
        file_list = [[[] for _ in range(self.nClass)] for _ in range(self.nbatch)]
        for batch_index in range(self.nbatch):  
            for flavour_index in range(self.nClass):
                for flavour, representation in self.Class.items():
                    if representation == flavour_index:
                        file_list[batch_index][flavour_index] = datasets[flavour][batch_index]
        
        return file_list 
    
    
     
    def __len__(self) -> int:
        #return total number of batches
        return self.nbatch 
         
    #def getNData(self) -> int:
        #return total number of jets 
    #    return self.batch_size * self.nClass * self.nbatch
     
    def __getitem__(self, index: int) -> Tuple:
        '''
        input 
        
            index: (int) index of batch
        
        output:
            return tensors gather as batch as a whole, pf, pf_var, hgcal, wgt. 
        ''' 
        return self.prepare_fn(self.filelabel_list[index], keys = self.npz_keys, shuffle = self.data_shuffle)



class JetClassOpenDataset(IterableDataset):
    """
    Customized dataset for loading JetClass Open dataset
    args:
    """
    pf_features = ['part_pt_log', 'part_e_log', 'part_ptrel', 'part_erel', 'part_deltaR', 'part_charge', 'part_isCHad', 'part_isNHad', 'part_isPhoton', 'part_isElectron', 'part_isMuon', 'part_deta', 'part_dphi']
    pf_vectors = [
        'part_px', 'part_py', 'part_pz', 'part_energy'
    ]
    
    def __init__(
        self,
        datasets_root: str = '/home/public/QuarkGluon-OpenData/',
        split: str = 'train',
        start: int =  0,
        end: int =  -1,
        prepare_fn: Callable = None,
        shuffle: bool = False,
        max_length: int = 64
    ) -> None:
        super(JetClassOpenDataset).__init__()
        
        self.filepaths = [os.path.join(datasets_root, fname) for fname in os.listdir(path = datasets_root) if split in fname]
        self.filepaths.sort()
        self.start = start
        self.end = end if end > 0 else len(self.filepaths)
        
        for filepath in self.filepaths:
            assert os.path.isfile(filepath)    
        
        if shuffle:
            self.filepaths = self.shuffled_data_list()
        self.prepare_fn = prepare_fn
        #assert self.prepare_fn is not None    
        #self.shuffle = shuffle
        self.max_length = max_length
        self.getNBatch()
    def process_data(self, filepath: str = None):
        table = pq.read_table(filepath) 

        yield from self.prepare_fn(table, pf_features = self.pf_features, max_length = self.max_length, pf_vectors = self.pf_vectors)
    def __chain__(self, filepaths: List[str]) -> chain:
        return chain.from_iterable(map(self.process_data, filepaths))  
        
    def get_stream(self, file_path) -> chain:
        
        Chain = chain.from_iterable(map(self.process_data, self.filepaths))  
        return Chain 
    
    def __len__(self) -> chain:
        """
        size of data
        """
        return self.data_size 
    def getNBatch(self) -> None:
        
        count = 0
        for filepath in self.filepaths:
            count += len(pq.read_table(filepath)['label'])
        
        self.data_size  = count
        
         
    def shuffled_data_list(self) -> List[str]:
        return random.sample(self.filepaths, len(self.filepaths))
    
     
    def __iter__(self) -> chain:
        worker_info = torch.utils.data.get_worker_info()
         
        if worker_info is None:
            iter_start = self.start
            iter_end = self.end
        else:
            per_worker = int(math.ceil((self.end - self.start) / float(worker_info.num_workers)))
            worker_id = worker_info.id
            iter_start = self.start + worker_id * per_worker
            iter_end = min(iter_start + per_worker, self.end)
        
        filepaths = self.filepaths[iter_start:iter_end]
        
        return self.__chain__(filepaths = filepaths)

def worker_init_fn(worker_init):
    set_seed(worker_init + 123)
    
    
        
     