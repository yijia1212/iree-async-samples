# Copyright 2023 The IREE Authors
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# An image that includes Tensorflow 2.12.0 for running on CPU.

# Ubuntu 22.04.
FROM ubuntu@sha256:817cfe4672284dcbfee885b1a66094fd907630d610cab329114d036716be49ba

######## Base ########
RUN apt-get update \
  && apt-get install -y \
    git \
    unzip \
    wget \
    curl \
    gnupg2 \
    python3-numpy

######## Python ########
WORKDIR /install-python

ARG PYTHON_VERSION=3.10

COPY python_build_requirements.txt install_python_deps.sh ./
RUN ./install_python_deps.sh "${PYTHON_VERSION}" \
  && apt-get -y install python-is-python3 \
  && rm -rf /install-python

ENV PYTHON_BIN /usr/bin/python3

WORKDIR /

######## Bazel ########
WORKDIR /install-bazel
COPY install_bazel.sh .bazelversion ./
RUN ./install_bazel.sh && rm -rf /install-bazel
WORKDIR /

######## Tensorflow ########
WORKDIR /tensorflow

# SHA where release 2.13.0-rc0 was cut.
ARG TENSORFLOW_COMMIT_SHA="525da8a"

COPY build_run_hlo_module.sh ./
RUN ./build_run_hlo_module.sh "${TENSORFLOW_COMMIT_SHA}"

ENV TF_RUN_HLO_MODULE_PATH="/tensorflow/tensorflow/bazel-bin/tensorflow/compiler/xla/tools/run_hlo_module"

WORKDIR /
