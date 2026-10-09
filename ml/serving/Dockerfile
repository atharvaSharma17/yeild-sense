FROM public.ecr.aws/docker/library/python:3.12-slim
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
RUN pip install --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir boto3
WORKDIR /app
COPY worker.py ./
RUN useradd --uid 10001 --create-home worker
USER worker
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 ML_LISTEN_ADDR=127.0.0.1:9000
CMD ["python", "worker.py"]
