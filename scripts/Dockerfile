FROM nvidia/cuda:10.1-devel
# Build ethminer from source so I can save money
# in between running experiments.
WORKDIR /
RUN apt update \
    && apt-get install -y \
        git \
        build-essential \
        cmake \
        mesa-common-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean
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

