#FROM ubuntu:18.04
FROM nvcr.io/nvidia/pytorch:21.02-py3
# Get dependencies
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    tmux ffmpeg libsm6 libxext6 git
RUN python3 -m pip install --upgrade pip \
    pip install wandb fvcore timm mmcv
