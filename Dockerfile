FROM pytorch/pytorch:1.7.0-cuda11.0-cudnn8-runtime
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY docker_entrypoint.sh .
RUN chmod +x ./docker_entrypoint.sh
COPY experiments experiments
COPY src src
CMD ["./docker_entrypoint.sh"]
