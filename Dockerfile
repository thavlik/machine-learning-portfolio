FROM rayproject/ray-ml:latest-gpu
USER root
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
RUN pip install awscli --force-reinstall --upgrade --ignore-installed
RUN pip install 'git+https://github.com/thavlik/nonechucks.git'
RUN pip install -r requirements.txt

WORKDIR /
RUN cd tmp \
    && wget https://developer.nvidia.com/compute/cuda/10.1/Prod/local_installers/cuda_10.1.105_418.39_linux.run \
    && sh ./cuda_10.1.105_418.39_linux.run \
    && rm ./cuda_10.1.105_418.39_linux.run

# Build ethminer from source so I can save money
# in between running experiments.
RUN git clone https://github.com/ethereum-mining/ethminer.git --branch release/0.17 \
    && cd ethminer \
    && git submodule update --init --recursive \
    && mkdir build \
    && cd build \
    && cmake \
        -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-10.1 \
        -D CUDA_NVCC_EXECUTABLE=/usr/local/cuda-10.1/bin/nvcc \
        -D CUDA_INCLUDE_DIRS=/usr/local/cuda-10.1/include \
        -D CUDA_CUDART_LIBRARY=/usr/local/cuda-10.1/lib64/libcudart.so \
        .. \
    && make install \
    && cd / \
    && rm -rf ethminer

#RUN cd /tmp \
#    && mkdir ethminer \
#    && cd ethminer \
#    && wget https://github.com/ethereum-mining/ethminer/releases/download/v0.18.0/ethminer-0.18.0-cuda-8-linux-x86_64.tar.gz \
#    && tar -xzvf ethminer-0.18.0-cuda-8-linux-x86_64.tar.gz \
#    && mv bin/ethminer /usr/local/bin/ethminer \
#    && cd .. \
#    && rm -rf ethminer

COPY scripts/mine-eth /usr/local/bin/mine-eth
RUN chmod +x /usr/local/bin/mine-eth

RUN git clone https://github.com/thavlik/machine-learning-portfolio.git
WORKDIR /machine-learning-portfolio
CMD ["./docker_entrypoint.sh"]
