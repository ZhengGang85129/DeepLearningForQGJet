# JetTagger
Deep Learning Model for Quark And Gluon Jet Tagging


## Setup

Suggest to use `miniconda` for python3 environment:


### Step1(Optional) : Download miniconda

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh # for Linux
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-MacOSX-arm64.sh # for Mac
sh Miniconda3-latest-Linux-x86_64.sh # for Linux
sh Miniconda3-latest-MacOSX-arm64.sh # for Mac
```

### Step2: Set `conda` environment

```bash
source $HOME/miniconda3/etc/profile.d/conda.sh
```

### Step3: Insall python3 modules

```bash
conda env create -f setup/enviroment.yml
```

### Step4: 
```bash
conda activate qgtagger_training # Enter conda
conda deactivate                 # Exit conda
```

### Training
```
python3 ./JetTagger/tool/train.py --configs ./experiments/train_config_mcnet.yaml # Please check the yaml file to ensure everything is adaptive to your local setting.
```

### Inference

```
python3 ./JetTagger/tool/test.py --configs ./experiments/train_config_mcnet.yaml # Please check `TEST` part in the yaml file to ensure checkpoint you want to verify is adaptive to your local setting.
```
