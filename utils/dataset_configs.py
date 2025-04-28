from JetTagger.utils.dataset import JetClassOpenDataset 
from JetTagger.utils.data_util import prepare_fn_OpenDataset, collate_fn_OpenDataset, prepare_fn_OpenDataset_test, collate_fn_OpenDataset_test 

DATASET_CONFIGS = {
    'JetClass_QuarkGluon': {
    'dataset_class': JetClassOpenDataset,
    'prepare_fn': prepare_fn_OpenDataset,
    'collate_fn': collate_fn_OpenDataset,
    'train_split': 'train',
    'val_split': 'val',
    }
}

TEST_DATASET_CONFIGS = {
    'JetClass_QuarkGluon': {
    'dataset_class': JetClassOpenDataset,
    'prepare_fn': prepare_fn_OpenDataset_test,
    'collate_fn': collate_fn_OpenDataset_test,
    'train_split': 'train',
    'val_split': 'val',
    }
}
