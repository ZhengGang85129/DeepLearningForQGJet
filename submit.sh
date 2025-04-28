#!/bin/bash
#SBATCH --job-name=simplescript
#SBATCH --output=logs/%x_%j.out
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4         # 給 dataloader 用
#SBATCH --gres=gpu:1              # 要求 1 張 GPU
#SBATCH --mem=16G
#SBATCH --time=06:00:00
#SBATCH --partition=v100-al9_short
cd /dicos_ui_home/zhenggang/ #/dicos_ui_home/zhenggang
source  /dicos_ui_home/zhenggang/miniconda3/etc/profile.d/conda.sh
conda activate qgtagger_training
pwd
nvidia-smi
#module list
python3 ./JetTagger/tool/train.py  --config JetTagger/configs/qg_jet_v9pt15/PoinT_final_Propagate.yaml #./JetTagger/configs/qg_jet_v9pt15/qg_jet_v9pt15_ParT_repro.yaml #JetTagger/configs/qg_jet_v9pt15/PoinT_final_Propagate.yaml
srun /bin/echo "Hello World!"

