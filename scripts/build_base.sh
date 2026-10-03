#!/usr/bin/env bash
# Build and push the base image (models + embeddings DB) to ACR, skipping it if this exact version already exists.
# Usage: scripts/build_base.sh <registry-name> [--wait]
set -euo pipefail

registry="${1:?Usage: scripts/build_base.sh <registry-name> [--wait]}"
cd "$(dirname "$0")/.."

inputs=(Dockerfile.base pyproject.toml uv.lock build_db.py embeddings.py operators.py vocab.py backend/__init__.py backend/db.py data/vocab_5000.txt)
tag="$(cat "${inputs[@]}" | shasum -a 256 | cut -c1-12)"
image="convergence-base:${tag}"

if az acr repository show-tags --name "$registry" --repository convergence-base --output tsv 2>/dev/null | grep -qx "$tag"; then
  echo "Base image ${registry}.azurecr.io/${image} already exists."
else
  wait_flag="--no-wait"
  [[ "${2:-}" == "--wait" ]] && wait_flag=""
  az acr build --registry "$registry" --image "$image" --file Dockerfile.base --timeout 7200 $wait_flag .
fi

echo "BASE_IMAGE=${registry}.azurecr.io/${image}"
