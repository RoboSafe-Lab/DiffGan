# Start to Use DiffGan
## Overview

This repository contains implementations for trajectory generation and reward learning using deep learning approaches.
    
### Components

- **MaxEntIRL**: Maximum Entropy Inverse Reinforcement Learning implementation for inferring reward feature weights using linear combinations of features.
- **DiffusionGan**: Diffusion Model-based trajectory generation system that creates candidate trajectories for MaxEntIRL, with the inferred rewards used to guide the diffusion process.

### Project Structure

```
DiffGan/
├── MaxEntIRL/                      # Maximum Entropy IRL implementation
├── DiffusionGan/                   # Diffusion-based trajectory generation
├── CTG/                
│   └── tbsim/                      # Core simulation and model framework
├── behavior-generation-dataset/    # Dataset
│   └── nuscenes/   
├── scripts/
├── sh/                             # dmog shell
├── config/                         # Configs for training and test
├── train_results/                  # Save checkpoints for Discriminator
└── checkpoints/                    # Save checkpoints for Generator (CTG)
```

## Setup
### Basic Installation (mainly based on tbsim)
Create conda environment
```bash
conda create -n bg3.9 python=3.9
conda activate bg3.9
```
Clone DiffGan (this repo)
```bash
git clone https://github.com/RoboSafe-Lab/DiffGan.git
cd DiffGan
```
Install `CTG`
```bash
cd CTG
pip install -e . --config-settings editable_mode=compat
```
Install a customized version of `trajdata`
```bash
cd ..
git clone https://github.com/AIasd/trajdata.git
cd trajdata
pip install -r trajdata_requirements.txt
pip install -e . --config-settings editable_mode=compat
```
Install `Pplan`
```bash
cd ..
git clone https://github.com/NVlabs/spline-planner.git Pplan
cd Pplan
pip install -e . --config-settings editable_mode=compat
```
Install `ffmpeg`:
```bash
cd ..
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xJf ffmpeg-release-amd64-static.tar.xz
```
### Potential Issue
One might need to run the following:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 torchmetrics==0.11.1 torchtext --extra-index-url https://download.pytorch.org/whl/cu113
```


## Quick Start
### Obtain Dataset
We currently support the [nuScenes](https://www.nuscenes.org/nuscenes) dataset.

* Download the nuScenes dataset (with the v1.3 map extension pack) and organize the dataset directory as follows:
    ```
    nuscenes/
    ├── maps/
    ├── v1.0-mini/
    └── v1.0-trainval/
    ```
### Before Training
* Use the examples in `trajdata` to preprocess the data:
    ```bash
    python trajdata/examples/preprocess_data.py
    ```
* Before training DiffGan, you need to follow [this](CTG/README.md) to train the diffuser model:
    ```bash
    python CTG/scripts/train.py --dataset_path <path-to-dataset> --config_name trajdata_nusc_diff --debug
    ```
    After that, you will find the trained weights in `diffuser_trained_models/`. Please move this folder to `checkpoints/` directory.

### Train DiffGan
```bash
python -m scripts.train --config=config/MaxEntIRL.json
```

### Inference DiffGan
```bash
python -m scripts.infer --config=config/MaxEntIRL.json
```

### Parse Metrics
```bash
python scripts/calcalate_metrics.py --hdf5_dir=<path-to-hdf5-file> --dataset_dir=<path-to-dataset> --output_dir=<path-to-output-dir>
```


## Pre-trained Models
We have provided checkpoints in `\train_results`. When using it, you need to modify the `pkl_label` parameter in the `config/config.json` file to `"best"`. 