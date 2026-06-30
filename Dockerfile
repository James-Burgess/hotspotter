# ── Build stage ──────────────────────────────────────────────────────────────
# Heavy: nvidia/cuda base, cmake, libopencv-dev, g++.  We only keep the
# compiled .so files and the Python virtualenv.
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu22.04 AS build

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8 DEBIAN_FRONTEND=noninteractive
ENV VENV=/opt/venv

RUN set -ex \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    ca-certificates build-essential pkg-config \
    python3 python3-dev python3-pip python3-setuptools python3-venv \
    libopencv-dev cmake libomp-dev liblz4-dev \
    libgdal-dev libeigen3-dev libgeos-dev libproj-dev \
    python-is-python3 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv "$VENV"
ENV PATH="$VENV/bin:$PATH"

RUN pip3 install --no-cache-dir \
    'setuptools>=69' 'pip>=23' cmake ninja scikit-build \
    'setuptools_scm[toml]' cython

COPY . /app
WORKDIR /app

# -- Compile sver (single-file g++) -------------------------------------------
RUN cd src/hotspotter/_vendor/sver/_sver_cpp \
    && g++ -shared -fPIC -O2 -ffast-math \
           -I/usr/include/opencv4 \
           sver.cpp -lopencv_core \
           -o libsver.so \
    && cd /app

# -- Compile hesaff (multi-file cmake) ----------------------------------------
ENV SETUPTOOLS_SCM_PRETEND_VERSION=4.0.0
RUN cd wbia-tpl-pyhesaff \
    && python3 setup.py build_ext --inplace \
    && mkdir -p /app/src/hotspotter/_vendor/pyhesaff/lib \
    && cp pyhesaff/lib/libhesaff*.so /app/src/hotspotter/_vendor/pyhesaff/lib/ \
    && cd ..

# -- Install pyflann from submodule --------------------------------------------
ENV SETUPTOOLS_SCM_PRETEND_VERSION=4.0.5.dev10
RUN cd wbia-tpl-pyflann \
    && pip3 install --no-cache-dir . \
    && find "$VENV/lib/python3.10/site-packages/pyflann/lib" \
         -name 'libflann.so.4.0.*' ! -name '*.a' -exec cp /app/vendor/libflann_wb.so {} \; \
    && find "$VENV/lib/python3.10/site-packages/pyflann/lib" \
         -name 'libflann.so*' ! -name '*.a' -exec cp /app/vendor/libflann_wb.so {} \;

# -- Install Python deps + hotspotter itself into venv -------------------------
RUN pip3 install --no-cache-dir \
    'numpy>=1.24,<2' \
    'opencv-contrib-python-headless==4.7.0.72' \
    'pydantic>=2.0' \
    'scipy>=1.10,<2' \
    faiss-cpu \
    annoy delorean gitpython lockfile matplotlib networkx \
    ordered-set pandas parse Pillow pint psutil pyarrow pyparsing \
    requests scikit-image scikit-learn statsmodels ubelt \
 && pip3 install --no-cache-dir --no-deps .


# ── Run stage ────────────────────────────────────────────────────────────────
# Slim: ubuntu 22.04 + only the runtime system libraries needed by the
# compiled extensions (libopencv_core / libgomp).  No cmake, no -dev headers,
# no CUDA bloat.
FROM ubuntu:22.04

ENV LC_ALL=C.UTF-8 LANG=C.UTF-8 DEBIAN_FRONTEND=noninteractive
ENV VENV=/opt/venv

RUN set -ex \
 && apt-get update \
 && apt-get install -y --no-install-recommends \
    ca-certificates \
    python3 python3-distutils \
    libopencv-core4.5d libopencv-imgproc4.5d libopencv-imgcodecs4.5d \
    libomp5-12 libgomp1 \
    liblz4-1 libgdal30 libgeos3.10.2 libproj22 \
    libjpeg8 libwebp7 libopenjp2-7 \
 && apt-get clean && rm -rf /var/lib/apt/lists/*

ENV PATH="$VENV/bin:$PATH"
ENV PYTHONPATH=/app/src
ENV LD_LIBRARY_PATH=/opt/venv/lib

COPY --from=build "$VENV" "$VENV"
COPY --from=build /app/tests /app/tests
COPY --from=build /app/scripts /app/scripts
COPY --from=build /app/pyproject.toml /app/pyproject.toml
COPY --from=build /app/src /app/src

WORKDIR /app
