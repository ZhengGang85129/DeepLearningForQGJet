import numpy as np
import torch, random, os
import yaml
import matplotlib.pyplot as plt
from typing import Dict, Union
import torch.backends.cudnn as cudnn


def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    cudnn.deterministic = True
    cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(f"Random seed set as {seed}")
    
    
class AverageMeter(object):
    """computes and stores the average and the current value"""
    def __init__(self):
        self.reset()
    
    def reset(self):
        self.current_value = 0
        self.avg_value = 0
        self.sum = 0
        self.count = 0 
    
    def update(self, value:float, weight:Union[int, float] = 1):
        self.current_value = value
        self.sum += self.current_value * weight
        self.count += weight 
        self.avg_value  = self.sum / self.count       
    
class Monitor(object):
    metrics = {
        '0': {
            'name': 'mLoss',
            'title': 'Averaged Cross Entropy',
            'y_lim': (0.4, 0.6)
        }, 
        '1': {
            'name': 'mAcc',
            'title': 'Averaged Accuracy',
            'y_lim': (0.7, 0.85)
        }
    }
    
    def __init__(self, save_path:str, resume = False) -> None:
        
        self.save_path = os.path.join(save_path, 'monitor')
        self.nmetric = len(self.metrics.items())
        self.resume = resume
    def reset(self) -> None:
        self.fig, self.ax = plt.subplots(1, self.nmetric, layout = "constrained") 

    def __plot(self) -> None:
        self.reset() 
        for index, (_, metric) in enumerate(self.metrics.items()):
            Y_train = self.state['train'][metric['name']]
            Y_evaltrain = self.state['eval-train'][metric['name']]
            Y_val = self.state['val'][metric['name']]
            Y_best_point = self.state['best_point'][metric['name']]
            X_best_point = self.state['best_point']['current_epoch'] 
            
            x = np.arange(1, len(Y_train) + 1)
            self.ax[index].plot(x, Y_train, label = 'train')
            self.ax[index].plot(x, Y_evaltrain, label = 'eval-train')
            self.ax[index].plot(x, Y_val, label = 'validation') 
            
            self.ax[index].set_ylim(*metric['y_lim'])
            self.ax[index].set_title(metric['title'])
            
            self.ax[index].axvline(
                x = X_best_point, 
                c = 'grey',
                linestyle = 'dashdot',
                alpha = 0.645
            )
            self.ax[index].axhline(
                y = Y_best_point, 
                c = 'grey',
                linestyle = 'dashdot',
                alpha = 0.645
            )
            self.ax[index].annotate(
                f'({X_best_point}, {Y_best_point:.3f})',
                (X_best_point, Y_best_point),
                textcoords = "offset points",
                xytext = (0, 5),
                ha = 'center',
                fontsize = 8
            )
            self.ax[index].scatter(X_best_point, Y_best_point, c = 'red', label = 'best point', s = 10)
            self.ax[index].legend(fontsize = 'small')
        self.fig.savefig(self.save_path + '.png') 
        self.fig.savefig(self.save_path + '.pdf') 
        return
    def __record(self) -> None:
        
        with open(self.save_path, 'w') as stream:
            yaml.safe_dump(self.state, stream)     
        
        return
    def update(self, state: Dict) -> None:
        self.state = state 
        
        self.__record()
        self.__plot()
         



       
        
        