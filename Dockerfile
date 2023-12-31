FROM python:3.10-slim-bullseye

WORKDIR /app

ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Etc/UTC
RUN apt-get update && apt-get install --no-install-recommends --no-install-suggests -y \
  curl \
  git \
  unzip \
  libgl1 \
  libglib2.0-0 \
  build-essential \
  libgoogle-perftools-dev \
  wget \
  libzbar0

# We need the latest pip
RUN pip install --upgrade --no-cache-dir pip


# Install dependencies
COPY requirements.txt .
ENV TORCH_CUDA_ARCH_LIST=All
ENV MAX_JOBS=4
ENV LD_PRELOAD=libtcmalloc.so
RUN pip install --upgrade --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu121
RUN pip install -v -U git+https://github.com/chengzeyi/stable-fast.git@main

COPY ./app ./

CMD ["python", "main.py"]