FROM public.ecr.aws/docker/library/python:3.13-slim
RUN pip install --no-cache-dir cfn-lint
ENTRYPOINT ["cfn-lint"]
