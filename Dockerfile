FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8

RUN set -ex \
 && apt-get update \
 && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    ca-certificates \
    build-essential \
    pkg-config \
    python3 \
    python3-dev \
    python3-pip \
    python3-setuptools \
    python3-venv \
    libopencv-dev \
    cmake \
    libboost-all-dev \
    libomp5 \
    git \
    libomp-dev \
    liblz4-dev \
    libgdal-dev \
    libeigen3-dev \
    libgeos-dev \
    libproj-dev \
    python-is-python3 \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir 'setuptools>=69' 'pip>=23' cmake ninja scikit-build 'setuptools_scm[toml]' cython

COPY . /app
WORKDIR /app

# Build toolkits in dependency order.  pyhesaff uses WBIA's build_ext --inplace
# workflow to match the compiled C++ output exactly.
ENV SETUPTOOLS_SCM_PRETEND_VERSION=4.0.6
RUN pip3 install --no-cache-dir --no-deps ./wbia-utool/
ENV SETUPTOOLS_SCM_PRETEND_VERSION=4.0.3
RUN pip3 install --no-cache-dir --no-deps ./wbia-vtool/
ENV SETUPTOOLS_SCM_PRETEND_VERSION=4.0.0
RUN cd wbia-tpl-pyhesaff \
    && python3 setup.py build_ext --inplace \
    && pip3 install --no-cache-dir --no-deps -e . \
    && cd ..

RUN pip3 install --no-cache-dir . flask

ENV LD_LIBRARY_PATH /virtualenv/env3/lib:${LD_LIBRARY_PATH}

EXPOSE 5000
ENTRYPOINT ["scripts/entrypoints/server-entrypoint.sh"]
