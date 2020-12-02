FROM rayproject/ray-ml:latest-gpu
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY docker_entrypoint.sh .
RUN chmod +x ./docker_entrypoint.sh
COPY experiments experiments
COPY src src
CMD ["./docker_entrypoint.sh"]
