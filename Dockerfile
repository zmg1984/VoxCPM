FROM nvidia/cuda:11.8.0-cudnn8-runtime-ubuntu22.04
WORKDIR /workspace
RUN apt update && apt install -y git python3 python3-pip python3-dev ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip3 install --upgrade pip && pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
COPY requirements.txt .
RUN pip3 install -r requirements.txt
COPY init.sh .
RUN chmod +x init.sh
CMD ["tail", "-f", "/dev/null"]
