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
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean \
    && npm install -g \
        electron@6.1.4 \
        orca \
        vtop
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

#RUN pip install 'git+https://github.com/facebookresearch/fvcore'
#ENV FORCE_CUDA="1"
#ARG TORCH_CUDA_ARCH_LIST="Kepler;Kepler+Tesla;Maxwell;Maxwell+Tegra;Pascal;Volta;Turing"
#ENV TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST}"
#RUN git clone https://github.com/facebookresearch/detectron2 detectron2_repo \
#    && pip install -e detectron2_repo \
#    && rm -rf detectron2_repo

COPY experiments experiments
COPY src src
COPY docker_entrypoint.sh .
RUN chmod +x ./docker_entrypoint.sh
CMD ["./docker_entrypoint.sh"]
