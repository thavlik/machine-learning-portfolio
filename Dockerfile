FROM rayproject/ray-ml:latest-gpu
RUN apt-get update \
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
        libdbus-1-dev \
        mesa-common-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && npm install -g \
        electron@6.1.4 \
        orca \
        vtop
RUN echo 'alias watchsmi="watch -n 0.5 nvidia-smi"' >> /root/.bashrc
WORKDIR /app
COPY requirements.txt .
RUN conda install cudatoolkit=10.1
RUN pip install https://download.pytorch.org/whl/cu101/torch-1.6.0%2Bcu101-cp37-cp37m-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cu101/torchvision-0.7.0%2Bcu101-cp37-cp37m-linux_x86_64.whl
RUN pip install awscli
RUN pip install 'git+https://github.com/thavlik/nonechucks.git'
RUN pip install -r requirements.txt

WORKDIR /
RUN git clone https://github.com/ethereum-mining/ethminer.git@release/0.17 \
    && cd ethminer \
    && git submodule update --init --recursive \
    && mkdir build \
    && cd build \
    && cmake ..
RUN make install
RUN git clone https://github.com/thavlik/machine-learning-portfolio.git

WORKDIR /machine-learning-portfolio
CMD ["./docker_entrypoint.sh"]
