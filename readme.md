conda create -n controlnet_service python=3.12

# 安装pip, 安装torch 12.8
pip3 install torch torchvision

# 安装 requirements.txt

# 安装detectron2,  https://detectron2.readthedocs.io/en/latest/tutorials/install.html
python -m pip install 'git+https://github.com/facebookresearch/detectron2.git'