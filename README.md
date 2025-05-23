# JetTagger



## Overview

<p align="center">
  <img src="https://github.com/user-attachments/assets/1cd2a69d-1209-4de4-bba0-62035b7c05fe" width="50%" />
  <img src="https://github.com/user-attachments/assets/337b5406-8398-4da3-bfbd-104f03e310fe" width="42%" height="95%" />
</p>


This project presents a deep learning framework for Quark and Gluon Jet classification, inspired by real-world challenges in high-energy physics experiments such as those at the CERN CMS detector.

We implement a custom Transformer model, named as MomentumCloudNet, with vector-based attention mechanisms and physics-informed features, achieving **state-of-the-art performance** on benchmark datasets. The model outperforms strong baselines including ParticleTransformer and ParticleNet, reaching an AUC of **0.9201** and accuracy of **0.849**.

## Model Details

<img width="908" alt="image" src="https://github.com/user-attachments/assets/ff31ea3a-8c77-4155-b0f5-74a981b5ac59" />




### Motivation
* Particle carries momentum vectors, intrinsic states (charge, particle type, etc)
* Existing particle transformer treat attention only as scalar weight, ignoring vectorial structure in interactions.

![image](https://github.com/user-attachments/assets/b6eb4865-797d-4c35-b68a-fb9a660f4f96)
### Proposed Features
  * Vector Attention Mechanism
  * Dynamic momentum update



## Installation Guide

We recommend using Miniconda to manage the Python3 environment.


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
python3 ./DeepLearningForQGJet/tool/train.py --configs ./experiments/template.yaml 
```
Note: Please check the YAML configuration file to ensure all settings are properly adapted to your local environment.

### Inference

```
python3 ./DeepLearningForQGJet/tool/test.py --configs ./experiments/template.yaml # Please check `TEST` part in the yaml file to ensure checkpoint you want to verify is adaptive to your local setting.
```
Note: Please verify the `TEST` section in the YAML file to ensure that the checkpoint path and related settings are correctly adpated to your local environment

## Performance

### Results
Our model demonstrates superior performance on the benchmark dataset for Jet classification.

|Model Name| AUC | Accuracy |
|:---------|:----|:---------|
|Our Model| 0.9201| 0.849|
|ParticleTransformer* | 0.9181 | 0.846|
|ParticleNet* | 0.9139 | 0.843 |
|LorentzNet* | 0.9156 | 0.844 |
 
* Note: model marked with an asterisk(*) are reported from their respective original papers.
* The improvement of +0.3% in accuracy and +0.2% in AUC over ParticleTransformer demonstrates the effectiveness of our vector-based attention mechanism and physics-motivated design.
### Computation Complexity
| Model Name | # parameters | MACs | Averaged inference time per event (run on single A100) |
|:-----------|:-------------|:-----|:-------------------------------------------------------|
| Our Model | 760 k | 1.42 B | 10.89 ms|
| ParticleTransformer | 2.14 M | 1.56 B | 9.24 ms|


### Model Interpretation (Attention strength)

<img width="815" alt="image" src="https://github.com/user-attachments/assets/7fce12b1-ca74-4e86-a99a-eefd65f242c6" />




### Model Interpretation (Attention Entropy)
<img width="644" alt="image" src="https://github.com/user-attachments/assets/b85fce39-7f91-4522-b52d-c2d07cd1d9e0" />

The distribution of attention entropy in different attention dimensions reflects the different patterns between quarks(cyan) and gluons(magenta). In most dimensions, quark jets show lower entropy(sharper attention).



## Contact
If you have any questions, suggestions, or collaboration ideas, feel free to reach out:
* Zheng-Gang Chen
  - Email: a0910555246@gmail.com
  - GitHub: [github.com/ZhengGang85129](https://github.com/ZhengGang85129)
* You-Ying Li
  - Email: you-ying.li@cern.ch
  - GitHub: [github.com/youyingli](https://github.com/youyingli)
 
  
Thank you for your interest in this project! Contributions and feedback are always welcome.


## References
- [1] Qu, H., et al. "Particle Transformer for Jet Tagging." *arXiv 2022*. [arXiv:2202.03772](https://arxiv.org/abs/2202.03772)
- [2] Qu, H., Gouskos, L. "ParticleNet: Jet Tagging via Particle Clouds." *arXiv 2020*. [arXiv:1902.08570](https://arxiv.org/abs/1902.08570)
- [3] Gong, D., et al. "LorentzNet: Lorentz Equivariant Graph Neural Network for Particle Physics." *arXiv 2022*. [arXiv:2206.13598](https://arxiv.org/abs/2206.13598)
- [4] Zhao,H., et al. "Point Transformer" *arXiv 2020*. [arXiv:2012.09164 ](https://arxiv.org/abs/2012.09164)

For Quark and Gluon open dataset, we use

```bibtex
@dataset{komiske_patrick_2019_3164691,
  author       = {Komiske, Patrick and
                  Metodiev, Eric and
                  Thaler, Jesse},
  title        = {Pythia8 Quark and Gluon Jets for Energy Flow},
  month        = may,
  year         = 2019,
  publisher    = {Zenodo},
  version      = {v1},
  doi          = {10.5281/zenodo.3164691},
  url          = {https://doi.org/10.5281/zenodo.3164691}
}
```

## Citation

If you use this codebase, please consider citing:

```bibtex
@misc{chen2024momentumcloudnet,
  author       = {Zheng-Gang Chen, You-Ying Li, Kai-Feng Chen},
  title        = {Quark and Gluon classification using Vector Attention},
  year         = {2024},
  howpublished = {\url{https://github.com/ZhengGang85129/JetTagger}},
  note         = {Work in progress}
}
