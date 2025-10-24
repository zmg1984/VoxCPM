#!/bin/bash
echo "===== 开始初始化VoxCPM环境 ====="

# 1. 检查GPU是否可用
echo "【1/3】检查GPU状态..."
nvidia-smi
if [ $? -eq 0 ]; then
  echo "GPU检查通过！"
else
  echo "警告：未检测到GPU，请确认平台已调度到GPU机器！"
fi

# 2. 安装系统依赖
echo "【2/3】安装系统依赖..."
apt update && apt install -y git python3-pip python3-dev ffmpeg --fix-missing

# 3. 安装Python依赖（匹配VoxCPM官方要求）
echo "【3/3】安装Python依赖..."
pip3 install --upgrade pip
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install -r requirements.txt

echo "===== 环境初始化完成！可以开始运行VoxCPM了～ ====="
