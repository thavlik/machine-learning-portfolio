FROM rayproject/ray-ml:latest-gpu
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY experiments experiments
COPY src src
COPY docker_entrypoint.sh .
RUN chmod +x ./docker_entrypoint.sh
CMD ["./docker_entrypoint.sh"]
