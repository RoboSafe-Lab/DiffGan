# DiffGAN
## Installation
### Basic (mainly based on tbsim)
Create conda environment
```bash
conda create -n bg3.9 python=3.9
conda activate bg3.9
```
Install `CTG (tbsim)`
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
One might need to run the following:
```bash
pip install torch==1.11.0+cu113 torchvision==0.12.0+cu113 torchaudio==0.11.0 torchmetrics==0.11.1 torchtext --extra-index-url https://download.pytorch.org/whl/cu113
```
Install ffmpeg:
```bash
wget https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz
```