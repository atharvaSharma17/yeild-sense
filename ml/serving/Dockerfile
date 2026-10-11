FROM public.ecr.aws/docker/library/python:3.12-slim
ARG TORCH_VERSION=2.13.0
ARG TORCHVISION_VERSION=0.28.0
RUN pip install --no-cache-dir torch==${TORCH_VERSION} torchvision==${TORCHVISION_VERSION} --index-url https://download.pytorch.org/whl/cpu
ARG SCIKIT_LEARN_VERSION=1.7.2
RUN pip install --no-cache-dir boto3 joblib scikit-learn==${SCIKIT_LEARN_VERSION}
WORKDIR /app
# Build with the repository root as context so both model architectures exist.
COPY ml/__init__.py /app/ml/__init__.py
COPY ml/model.py /app/ml/model.py
COPY ml/track_a/ /app/ml/track_a/
COPY ml/serving/worker.py /app/ml/serving/worker.py
RUN useradd --uid 10001 --create-home worker
USER worker
ENV PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 ML_LISTEN_ADDR=127.0.0.1:9000
CMD ["python", "ml/serving/worker.py"]
