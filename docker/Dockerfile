FROM pytorch/pytorch:2.3.0-cuda12.1-cudnn8-devel

# For the convenience for users in China mainland
# COPY apt-sources.list /etc/apt/sources.list

# Install some basic utilities
RUN chmod 1777 /tmp
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    sudo \
    git \
    bzip2 \
    libx11-6 \
    gcc \
    g++ \
    libusb-1.0-0 \
    libgl1-mesa-glx \
    libglib2.0-dev \
    openssh-server \
    openssh-client \
    iputils-ping \
    unzip \
    cmake \
    libssl-dev \
    libosmesa6-dev \
    freeglut3-dev \
    ffmpeg \
    iputils-ping \
 && rm -rf /var/lib/apt/lists/*

# # For the convenience for users in China mainland
# RUN pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple \
#   && export PATH="/usr/local/bin:$PATH" \
#   && /bin/bash -c "source ~/.bashrc"
# RUN conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/ \
#  && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/main/ \
#  && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/ \
#  && conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/ \
#  && conda config --set show_channel_urls yes

# Install dependencies
COPY requirements.txt requirements.txt 
RUN pip install -r requirements.txt --user 
RUN pip install -U xformers==0.0.26.post1 --index-url https://download.pytorch.org/whl/cu121

CMD ["python3"]