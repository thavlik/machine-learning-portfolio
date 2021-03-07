# Build ethminer from source so I can save money in between experiments.
FROM thavlik/ethminer:latest AS ethminer
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
RUN conda install cudatoolkit=10.1
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
