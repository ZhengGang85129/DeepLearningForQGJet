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

import awkward_pandas 


class UniJetDataset(IterableDataset):
    Class = {'g': 0, 'light': 1}
    batch_size = 256
    p4_features = ['Part_px', 'Part_py', 'Part_pz', 'Part_energy']
    p_features = ['Part_rel_pt', 'Part_rel_energy', 'Part_deltaEta', 'Part_deltaPhi', 'Part_charge', 'Part_isEle', 'Part_isMu', 'Part_isGamma', 'Part_isChargedHad', 'Part_isNeutralHad', 'Part_tanhd0', 'Part_tanhdz', 'Part_sigmad0', 'Part_sigmadz']
    
    def __init__(
        self, 
        datasets_root: str = '/home/public',
        split: str = 'val',
        start: int = 0,
        end: int = -1,
        prepare_fn: Callable = None,
        shuffle: bool = False,
        max_length: int = 64) -> None: 
        """
        max_length (int): Maximum number of particles per event.
        """
        super(UniJetDataset).__init__()
        
        self.datasets_root = os.path.join(datasets_root, split) 
        
        self.filepaths = []
        for CLASS in self.Class.keys():
            dir_name = os.path.join(self.datasets_root, CLASS) 
            for fname in os.listdir(dir_name):
                self.filepaths.append(os.path.join(dir_name, fname))
        self.filepaths.sort()
        self.start = start
        self.end = end if end > 0 else len(self.filepaths)
        
        self.nbatch = len(self.filepaths) 
        self.prepare_fn = prepare_fn
        #assert self.prepare_fn is not None    
        #self.shuffle = shuffle
        self.max_length = max_length
        self.getNBatch()
    def process_data(self, filepath: str = None):
        #table = pq.read_table(filepath) 
        table = awkward_pandas.read_parquet(filepath)
        
        yield from self.prepare_fn(table, p4_features = self.p4_features, max_length = self.max_length, p_features = self.p_features)
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
            count += len(pq.read_table(filepath)['Label'])
        
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
        
if __name__ == "__main__":
    A = UniJetDataset(split = 'val') 
        
        
        
