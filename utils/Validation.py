import os
import torch
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

#------------------------------------------------------------
# Output directory check and create
#------------------------------------------------------------
def _has_directory(path: str):
    if not os.path.exists(path):
        os.system(f'mkdir -p {path}')

#------------------------------------------------------------
# Loss/Learning rate function monitor
#------------------------------------------------------------
def make_loss_function(loss_set: dict, filename='loss', outdir='./', is_log=False):

    _has_directory(outdir)

    fig, ax = plt.subplots()

    for name, loss in loss_set.items():
        iteration = np.arange(1, len(loss)+1, 1)
        ax.plot(iteration, loss, label=name)
        np.savez(f'{outdir}/{filename}_{name}.npz', loss=loss)

    ax.legend()
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Loss')
    if is_log:
        ax.set_yscale('log')
    ax.set_title('Loss')
    plt.savefig(f'{outdir}/{filename}.png')
    plt.savefig(f'{outdir}/{filename}.pdf')
    plt.close(fig)

#------------------------------------------------------------
# ROC curve
#------------------------------------------------------------
def make_ROC_curve(information: dict, filename='roc_curve', outdir='./', isRaw=True):

    _has_directory(outdir)

    fig, ax = plt.subplots()
    ax.set_title('Receiver Operating Characteristic')

    roc_auc = -1.

    for name, info in information.items():

        if isRaw:

            y_target, probability, weight = info

            fpr, tpr, _ = metrics.roc_curve(y_target, probability, pos_label=1, sample_weight=weight)
            roc_auc = metrics.auc(fpr, tpr)
            rfpr = 1. - fpr

            ax.plot(tpr, rfpr, label = f"AUC ({name}) = {roc_auc:>3f}")

            np.savez(f"{outdir}/roc_curve_{name}.npz", tpr=tpr, rfpr=rfpr, auc=np.array([roc_auc], dtype=np.float32))
        else:

            roc_auc = info['auc'][0]
            ax.plot(info['tpr'], info['rfpr'], label = f"AUC ({name}) = {roc_auc:.3f}")

    ax.legend(loc = 'lower left')
    ax.plot([0, 1], [1, 0],'--')
    ax.set_ylabel('Signal efficiency')
    ax.set_xlabel('Background reduction')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    plt.savefig(f'{outdir}/{filename}.png')
    plt.savefig(f'{outdir}/{filename}.pdf')
    plt.close(fig)

    return roc_auc

#------------------------------------------------------------
# Probability histogram
#------------------------------------------------------------
def make_probability(probability:dict, filename='hist', outdir='./'):

    _has_directory(outdir)

    fig, ax = plt.subplots()

    for name, (prob, weights) in probability.items():
        ax.hist( prob , bins=60, label=f'{name}', histtype='step', weights=weights )

    ax.legend(loc='upper right')
    plt.savefig(f'{outdir}/{filename}.png')
    plt.savefig(f'{outdir}/{filename}.pdf')
    plt.close(fig)

#------------------------------------------------------------
# Reveal the auc as a function of a given indicator (split_variab)
#------------------------------------------------------------
@torch.no_grad()
def performance_reveal(model, dataset, outdir: str, split_variable: str, split_value: list) ->None:

    indicator = []
    y_pred    = []
    y_target  = []
    weight    = []

    model.eval()
    for var, X, X_sup, y, wgt in dataset:
        indicator .append(var)
        y_pred    .append(model(X)[:,1])
        y_target  .append(y)
        weight    .append(wgt)

    indicator  = torch.cat( indicator  , dim=0).to(torch.device('cpu')).numpy()
    y_pred     = torch.cat( y_pred     , dim=0).to(torch.device('cpu')).numpy()
    y_target   = torch.cat( y_target   , dim=0).to(torch.device('cpu')).numpy()
    weight     = torch.cat( weight     , dim=0).to(torch.device('cpu')).numpy()

    # Overall ROC curve
    overall_auc = make_ROC_curve(  { f'overall' : (y_target, y_pred, weight) }, 'roc_curve_overall', outdir )

    # Extra check probability
    prob_dict = { 'Gluon' : (y_pred[ y_target < 0.5 ], weight[ y_target < 0.5 ]), 'Quark' : (y_pred[ y_target > 0.5 ], weight[ y_target > 0.5 ]) }
    make_probability (prob_dict, 'overall_hist', outdir)

    # ROC curve with indicator splitting
    roc_auc = []

    for i in range(len(split_value)-1):

        cut = ( indicator > split_value[i] ) & ( indicator < split_value[i+1] )

        wgt = weight[cut]
        y   = y_target[cut]
        y_p = y_pred[cut]

        scale = np.sum(wgt[y>0.5]) / np.sum(wgt[y<0.5])
        wgt[y<0.5] *= scale

        roc_auc.append( make_ROC_curve(  { f'{split_variable}_{split_value[i]}_{split_value[i+1]}' : (y, y_p, wgt) }, f'roc_curve_{split_variable}_{split_value[i]}_{split_value[i+1]}', outdir ) )

        prob_dict = { 'Gluon' : (y_p[ y < 0.5 ], wgt[ y < 0.5 ]), 'Quark' : (y_p[ y > 0.5 ], wgt[ y > 0.5 ]) }
        make_probability (prob_dict, f'hist_{split_variable}_{split_value[i]}_{split_value[i+1]}', outdir)


    # AUC value as a function of the given indicator
    split_value = np.array(split_value)
    var   = (split_value[:-1] + split_value[1:]) * 0.5
    error = (split_value[1:] - split_value[:-1]) * 0.5

    fig, ax = plt.subplots()
    ax.errorbar(var, np.array(roc_auc), xerr=error, fmt='o', label=f"Overall AUC = {overall_auc:>3f}")

    ax.legend()
    ax.set_xlabel(split_variable)
    ax.set_ylabel('AUC')
    plt.savefig(f'{outdir}/auc_{split_variable}_split.png')
    plt.savefig(f'{outdir}/auc_{split_variable}_split.pdf')
    np.savez(f"{outdir}/auc_{split_variable}_split.npz", var=var, error=error, overall_auc=np.array(overall_auc), auc=np.array(roc_auc))
    plt.close(fig)

#------------------------------------------------------------
# Comparision the auc as a function of a given indicator (split_variable)
# between different situation
#------------------------------------------------------------
def make_comparison(information: dict, xaixs: str, outdir: str):

    _has_directory(outdir)

    fig, ax = plt.subplots()

    for name, path in information.items():
        auc = []

        var = None
        var_error = None
        overall_auc = []

        for dirname in os.listdir(path):

            npz_name = path + '/' + dirname + '/performance/'

            var = np.load(npz_name + 'auc_pt_split.npz')['var']
            var_error = np.load(npz_name + 'auc_pt_split.npz')['error']
            auc.append( np.expand_dims( np.load(npz_name + 'auc_pt_split.npz')['auc'], axis=1 ) )
            overall_auc.append( np.load(npz_name + 'roc_curve_overall.npz')['auc'] )

        auc = np.concatenate(auc, axis=1)
        overall_auc = np.concatenate(overall_auc)

        auc_mean = np.mean( auc, axis=1 )
        auc_std  = np.std( auc, axis=1 )
        overall_auc = overall_auc.mean()

        ax.errorbar(var, auc_mean, xerr=var_error, yerr=auc_std, fmt='.', label=f"{name} (Overall AUC = {overall_auc:>3f})")

    ax.legend(loc = 'lower right')
    ax.set_ylabel('AUC')
    ax.set_xlabel(xaixs)
    #ax.set_yscale('log')
    #ax.set_xlim(0, 1)
    #ax.set_ylim(0, 1)
    plt.savefig(f'{outdir}/co.png')
    plt.savefig(f'{outdir}/co.pdf')
    plt.close(fig)
