#!/bin/bash
echo "===== 开始初始化VoxCPM环境 ====="
nvidia-smi || echo "NVIDIA驱动未检测到，尝试继续..."
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu117
pip3 install -r requirements.txt
echo "===== 环境初始化完成！ ====="
