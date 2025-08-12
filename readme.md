conda create -n controlnet_service python=3.12

# 1.安装 torch
## 安装pip, 安装torch 12.8
pip3 install torch torchvision

## 或者 安装 CUDA 12.6
# pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --index-url https://download.pytorch.org/whl/cu126

# 2. 安装 requirements.txt


# 3. 安装detectron2,  https://detectron2.readthedocs.io/en/latest/tutorials/install.html
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'