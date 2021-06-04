FROM nvidia/cuda:10.1-cudnn7-runtime-ubuntu18.04

RUN apt update && \
    apt install -y bash \
                   build-essential \
                   git \
                   curl \
                   ca-certificates \
                   python3 \
                   python3-pip && \
    rm -rf /var/lib/apt/lists

RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir \
    mkl \
    torch \
    tornado \
    tensorboard \
    spacy \
    sklearn

RUN python3 -m pip install --no-cache-dir \
    boto3 \
    regex \
    sacremoses

RUN python3 -m spacy download en 

RUN mkdir /workspace
COPY . /workspace/usr/
WORKDIR /workspace/usr

EXPOSE 8888

ENTRYPOINT ["/bin/bash", "bootstrap.sh"]