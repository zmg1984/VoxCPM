#!/bin/bash
echo "===== 开始初始化VoxCPM环境 ====="
nvidia-smi
apt update && apt install -y git python3-pip python3-dev ffmpeg --fix-missing
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt
echo "===== 环境初始化完成！ ====="
