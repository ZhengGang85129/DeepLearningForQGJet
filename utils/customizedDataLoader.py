import os
import random

# Move tensor(s) to chosen device
def to_device(data, device):
    if isinstance(data, (list,tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)

# Make iterable dataloader with device
class ModifiedDataLoader():

    def __init__(self, dataloader, device):

        self.dl = dataloader
        self.device = device

    def __len__(self):
        return len(self.dl)

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def getNData(self):
        return len(self.dl.dataset)

class MultiDataLoader():

    '''
        Batch size must be 256 for each class. If not, should modify this code
    '''

    def __init__(self, dataset_dict, nbatch, func_prepare, device, nsplit : int = -1):

        self.nbatch = self._nbatch_maker(dataset_dict, nbatch)
        self.nClass = len(dataset_dict)
        self.filelist = self._filelist_maker(dataset_dict)
        self.prepare = func_prepare
        self.device = device
        self.nsplit = nsplit if nsplit > 1 else 1

    def _nbatch_maker(self, dataset_dict, nbatch):

        min_nfile = min( [ len(os.listdir(path)) for path, _ in dataset_dict.items() ] )
        if nbatch < 1:
            print(f'Auto-mode : assign nbatch to {min_nfile} which is the minimum of the number of files in each class')
            return min_nfile
        elif min_nfile > nbatch:
            return nbatch
        else:
            print(f'The nbatch={nbatch} you set is more than the minimum of the number of files in each class which is {min_nfile}. Automatically assign nbatch to {min_nfile}')
            return min_nfile

    def _filelist_maker(self, dataset_dict):

        filelist = []
        for path, target in dataset_dict.items():
            filelist += [ (path + '/' + ifile, target) for ifile in os.listdir(path) ][0:self.nbatch]

        # Rearrange the order of dataset to [ [class1, class2,...], [class1, class2,...], ... ]
        filelist = [ [filelist[i+  self.nbatch*j] for j in range(self.nClass)] for i in range(self.nbatch) ]

        return filelist

    def __len__(self):
        return self.nbatch * self.nsplit

    def __iter__(self):

        for file in self.filelist:
            for i_split in range(self.nsplit):
                yield self.prepare(file, self.device, split=(i_split, self.nsplit))

    def getNData(self):
        return self.nbatch * self.nClass * 512

    def shuffle(self):
        random.shuffle(self.filelist)
