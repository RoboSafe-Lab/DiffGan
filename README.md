<div align="center">
  <h1>DiffGAN</h1>
  <h2>GAN-Inspired Trajectory Generation by Inverse Reinforcement Learning-Informed Diffusion</h2>
</div>

## Table of Contents
- [Methods](#methods)
- [Simulation Results](#simulation-results)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)

## Methods
DiffGAN is a novel generative adversarial diffusion framework that combines diffusion models with inverse reinforcement learning (IRL) for realistic and diverse agent behavior modeling in autonomous vehicle simulation.

### Core Architecture
- **Generator:** Diffusion Model-based trajectory generation system that creates candidate trajectories for MaxEntIRL, with the inferred rewards used to guide the diffusion process.
- **Discriminator:** Maximum Entropy Inverse Reinforcement Learning implementation for inferring reward feature weights using linear combinations of features.

### Advantages
- Bridges the gap between discriminative learning (high realism) and generative learning (multimodal diversity).
- Produces agent behaviors that are both human-like and diverse

<a href="images/Diffgan.png">
  <img src="images/Diffgan.png" width="100%">
</a>

## Simulation Results
<details open>
  <summary><strong>Simulation on nuScenes-Boston</strong></summary>
  <br>
  <p align="center">
  <a href="images/b36.gif">
    <img src="images/b36.gif" width="100%">
  </a>
  <a href="images/b74.gif">
    <img src="images/b74.gif" width="100%">
  </a>
  <a href="images/b52.gif">
    <img src="images/b52.gif" width="100%">
  </a>
  </p>
</details>
<details open>
  <summary><strong>Simulation on nuScenes-Singapore</strong></summary>
  <br>
  <p align="center">
  <a href="images/s13.gif">
    <img src="images/s13.gif" width="100%">
  </a>
  <a href="images/s30.gif">
    <img src="images/s30.gif" width="100%">
  </a>
  <a href="images/s24.gif">
    <img src="images/s24.gif" width="100%">
  </a>
  </p>
</details>

## Project Structure
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

## Getting Started
### Basic Installation (mainly based on tbsim)
Create conda environment
```bash
conda create -n bg3.9 python=3.9
conda activate bg3.9

# Clone DiffGan (this repo)
git clone https://github.com/RoboSafe-Lab/DiffGan.git
cd DiffGan

# Install CTG
cd CTG
pip install -e . --config-settings editable_mode=compat

# Install a customized version of trajdata
cd ..
git clone https://github.com/AIasd/trajdata.git
cd trajdata
pip install -r trajdata_requirements.txt
pip install -e . --config-settings editable_mode=compat

# Install Pplan
cd ..
git clone https://github.com/NVlabs/spline-planner.git Pplan
cd Pplan
pip install -e . --config-settings editable_mode=compat

# Install ffmpeg
cd ..
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
tar -xJf ffmpeg-release-amd64-static.tar.xz
```
### Potential Issue
One might need to run the following:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 torchmetrics==0.11.1 torchtext --extra-index-url https://download.pytorch.org/whl/cu113
```

### Obtain Dataset
We currently support the [nuScenes](https://www.nuscenes.org/nuscenes) dataset.
* Download the nuScenes dataset and organize the dataset directory as follows:
    ```
    nuscenes/
    ├── maps/
    ├── v1.0-mini/
    └── v1.0-trainval/
    ```
* Use the examples in `trajdata` to preprocess the data:
    ```bash
    python trajdata/examples/preprocess_data.py
    ```
    
### Train Diffusion Model    
follow [this](CTG/README.md) to train the diffuser model:
```bash
python CTG/scripts/train.py --dataset_path <path-to-dataset> --config_name trajdata_nusc_diff --debug
```
After that, you will find the trained weights in `diffuser_trained_models/`. Please move this folder to `checkpoints/` directory.

### Train DiffGan
Before start training, ensure that you get the checkpoint of diffusion model and put it in the correct place. Please check `config/MaxEntIRL.json` to match `policy_ckpt_dir` and `policy_ckpt_key`
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










