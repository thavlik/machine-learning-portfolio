# Build ethminer from source so I can save money in between experiments.
ARG CUDA_VERSION=10.1
FROM nvidia/cuda:${CUDA_VERSION}-devel AS ethminer
ENV CUDA_VERSION=CUDA_VERSION
WORKDIR /
RUN apt update \
    && apt-get install -y \
        git \
        build-essential \
        cmake \
        mesa-common-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
RUN git clone https://github.com/ethereum-mining/ethminer.git --branch release/0.17 \
    && cd ethminer \
    && git submodule update --init --recursive \
    && mkdir build \
    && cd build \
    && cmake \
        -D CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda-${CUDA_VERSION} \
        -D CUDA_NVCC_EXECUTABLE=/usr/local/cuda-${CUDA_VERSION}/bin/nvcc \
        -D CUDA_INCLUDE_DIRS=/usr/local/cuda-${CUDA_VERSION}/include \
        -D CUDA_CUDART_LIBRARY=/usr/local/cuda-${CUDA_VERSION}/lib64/libcudart.so \
        .. \
    && make install \
    && cd / \
    && rm -rf ethminer

FROM rayproject/ray-ml:latest-gpu
USER root
ENV CUDA_VERSION=${CUDA_VERSION}
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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && npm install -g \
        electron@6.1.4 \
        orca \
        vtop
RUN echo 'alias watchsmi="watch -n 0.5 nvidia-smi"' >> /root/.bashrc
COPY --from=ethminer /usr/local/bin/ethminer /usr/local/bin/ethminer
WORKDIR /app
COPY requirements.txt .
RUN conda install cudatoolkit=${CUDA_VERSION}
RUN pip install https://download.pytorch.org/whl/cu101/torch-1.6.0%2Bcu101-cp37-cp37m-linux_x86_64.whl
RUN pip install https://download.pytorch.org/whl/cu101/torchvision-0.7.0%2Bcu101-cp37-cp37m-linux_x86_64.whl
RUN pip install awscli --force-reinstall --upgrade --ignore-installed
RUN pip install 'git+https://github.com/thavlik/nonechucks.git'
RUN pip install -r requirements.txt
COPY scripts/mine-eth /usr/local/bin/mine-eth
RUN chmod +x /usr/local/bin/mine-eth
RUN git clone https://github.com/thavlik/machine-learning-portfolio.git
WORKDIR /machine-learning-portfolio
CMD ["./docker_entrypoint.sh"]
