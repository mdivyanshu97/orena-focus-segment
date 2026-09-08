#!/usr/bin/env bash

set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DOCKER_IMAGE_TAG="segment-algorithm"
MODEL_PATH="${SCRIPT_DIR}/resources/base/model.safetensors"

if [ ! -f "$MODEL_PATH" ]; then
  echo "Missing ${MODEL_PATH}"
  echo "Run: python scripts/download_weights.py"
  exit 1
fi

docker build \
  --platform=linux/amd64 \
  --tag "$DOCKER_IMAGE_TAG"  \
  "$SCRIPT_DIR" 2>&1
