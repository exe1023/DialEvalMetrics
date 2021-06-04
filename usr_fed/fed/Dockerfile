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
    transformers \
    tornado

RUN mkdir /.transformer_cache
RUN chmod 777 /.transformer_cache
ENV PYTORCH_TRANSFORMERS_CACHE /.transformer_cache
RUN python3 -c 'import os; from transformers import AutoTokenizer; AutoTokenizer.from_pretrained("microsoft/DialoGPT-large", cache_dir = os.getenv("PYTORCH_TRANSFORMERS_CACHE", ""))'
RUN python3 -c 'import os; from transformers import AutoModelWithLMHead; AutoModelWithLMHead.from_pretrained("microsoft/DialoGPT-large", cache_dir = os.getenv("PYTORCH_TRANSFORMERS_CACHE", ""))'

RUN mkdir /workspace
COPY . /workspace/fed/
WORKDIR /workspace/fed/

EXPOSE 8888

ENTRYPOINT ["/bin/bash", "bootstrap.sh"]