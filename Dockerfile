# 基础镜像：带GPU的Ubuntu+Python环境（匹配CUDA 11.8）
FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04

# 设置工作目录（与CNB在线IDE的/workspace对应）
WORKDIR /workspace

# 安装基础工具
RUN apt update && apt install -y \
    git \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# 升级pip并安装PyTorch（匹配CUDA版本）
RUN pip3 install --upgrade pip && \
    pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 复制依赖清单
COPY requirements.txt .

# 安装项目依赖
RUN pip3 install -r requirements.txt

# 复制初始化脚本
COPY init.sh .

# 赋予脚本执行权限
RUN chmod +x init.sh

# 保持容器运行（供CNB IDE连接）
CMD ["tail", "-f", "/dev/null"]
