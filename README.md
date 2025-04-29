# JetTagger
Deep Learning Model for Quark And Gluon Jet Tagging


## Installation Guide

We recommend using Miniconda to manage the Python3 enviroment.


### Step 1(Optional) : Install Miniconda
For Linux:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh # for Linux
sh Miniconda3-latest-Linux-x86_64.sh # for Linux
```
For Mac:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh # for Mac
sh Miniconda3-latest-MacOSX-arm64.sh # for Mac
```

### Step 2: Initialize Conda

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
```

### Step 3: Create the Conda Environment

```bash
conda env create -f setup/enviroment.yml
```

### Step 4: Activate the Environment
```bash
conda activate qgtagger_training # Enter conda
conda deactivate                 # Exit conda
```
## Usage 
### Training
```
python3 ./JetTagger/tool/train.py --configs ./experiments/template.yaml 
```
Note: Please check the YAML configuration file to ensure all settings are properly adapted to your local environment.

### Inference

```
python3 ./JetTagger/tool/test.py --configs ./experiments/train_config_mcnet.yaml # Please check `TEST` part in the yaml file to ensure checkpoint you want to verify is adaptive to your local setting.
```
Note: Please verify the `TEST` section in the YAML file to ensure that the checkpoint path and related settings are correctly adpated to your local environment
