FROM rayproject/ray-ml:latest-gpu
USER root
RUN apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub \
    && apt-get update \
    && apt-get install -y \
        chromium-browser \
        fonts-liberation \
        xvfb \
        poppler-utils \
        libxss1 \
        libnss3 \
        libnss3-dev \
        libgdk-pixbuf2.0-dev \
        libgtk-3-dev \
        libxss-dev \
        libasound2 \
        libgtk2.0-dev \
        zlib1g-dev \
        libgl1-mesa-dev \
        nodejs \
        npm \
        nano \
        htop \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && npm install -g \
        orca \
        vtop
RUN echo 'alias watchsmi="watch -n 0.5 nvidia-smi"' >> /root/.bashrc
WORKDIR /app
COPY requirements.txt .
#RUN conda install cudatoolkit=10.1
RUN pip install torch==1.7.1+cu110 torchvision==0.8.2+cu110 torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html
#RUN pip install https://download.pytorch.org/whl/cu111/torch-1.7.1%2Bcu101-cp37-cp37m-linux_x86_64.whl
#RUN pip install https://download.pytorch.org/whl/cu111/torchvision-0.8.1%2Bcu101-cp37-cp37m-linux_x86_64.whl
RUN pip install awscli --force-reinstall --upgrade --ignore-installed
RUN pip install 'git+https://github.com/thavlik/nonechucks.git'
RUN pip install -r requirements.txt
WORKDIR /machine-learning-portfolio
COPY . .
CMD ["./docker_entrypoint.sh"]
