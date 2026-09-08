FROM --platform=linux/amd64 pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime AS segment-algorithm-amd64

# torch 2.11.0 + torchvision are pre-installed in the base image (CUDA 12.8 — required for the platform's RTX PRO 6000 GPUs; the original 2.5.1/cu12.4 base cannot run on them, per the organizers' 2026-08-05 forum post).
# requirements.txt only lists additional dependencies.

ENV PYTHONUNBUFFERED=1

# ── System packages ───────────────────────────────────────────────────────────
# The slim -runtime base image ships neither the shared libraries that
# opencv-python (a dependency of orena-focus) needs at import time, nor any
# compiler toolchain. Install system packages HERE, while the build still runs
# as root — everything below runs as the unprivileged 'user'. If one of your
# pip dependencies compiles from source (e.g. flash-attn), add build-essential.
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
        libxcb1 \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user requirements.txt /opt/app/

# --break-system-packages: the 2.10.0 base ships a PEP-668 "externally managed"
# Debian Python; in a single-purpose container the guard protects nothing.
RUN python -m pip install \
    --break-system-packages \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# ── Guard (from upstream 64a264c): torch must still be the CUDA 12.8 build ────
# If pip swapped torch for the default PyPI wheel (CUDA 13.0), the image cannot
# start on the L40S fleet (driver 570 < required 580) and every platform job
# FAILS with no visible log. The CUDA version is not visible in the arch list, so
# both halves are asserted. get_arch_list() is empty with no GPU (docker build),
# hence the compiled arch flags are read directly.
RUN python -c "import torch; \
flags = torch._C._cuda_getArchFlags() or ''; \
cuda = torch.version.cuda or ''; \
print('torch', torch.__version__, '| cuda', cuda, '|', flags); \
assert cuda.startswith('12.8'), 'torch was replaced by a CUDA ' + cuda + ' build; the L40S driver cannot run it'; \
assert 'sm_120' in flags and 'sm_86' in flags, 'torch build misses a required GPU architecture: ' + repr(flags)"

# ── Model definition + weights ────────────────────────────────────────────────
# resources/base/ must contain the public merged checkpoint. Populate it with
# `python scripts/download_weights.py` before building.
#
# ECR layer-size limit: a single image layer must not exceed 50 GB, and each
# COPY instruction produces exactly one layer. A checkpoint large enough to push
# one COPY past that limit must be split into chunks and copied with several COPY
# instructions (one layer each), then reassembled at runtime — e.g. pre-split
# with `split -b 45G weights.pt resources/weights.part-` and:
#     COPY --chown=user:user resources/weights.part-aa /opt/app/resources/
#     COPY --chown=user:user resources/weights.part-ab /opt/app/resources/
#     COPY --chown=user:user resources/weights.part-ac /opt/app/resources/
# reassembling with `cat resources/weights.part-* > weights.pt` before load.
COPY --chown=user:user resources/ /opt/app/resources/

COPY --chown=user:user inference.py /opt/app/

ENTRYPOINT ["python", "inference.py"]
