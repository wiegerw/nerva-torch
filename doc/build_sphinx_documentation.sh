#!/bin/bash
set -euo pipefail

# Resolve repository root and script directory (so it works from any CWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Output directory at the repository root (so CI publishes ./output_directory)
OUTPUT_DIRECTORY="${REPO_ROOT}/output_directory"
mkdir -p "${OUTPUT_DIRECTORY}"

# Ensure Sphinx + theme are available (only for local builds;
# in CI this step is handled in the workflow)
python3 -m pip install --upgrade sphinx sphinx-rtd-theme

# Build HTML docs from doc/sphinx into output_directory/sphinx
sphinx-build -b html "${REPO_ROOT}/doc/sphinx" "${OUTPUT_DIRECTORY}/sphinx"

echo "The Sphinx HTML docs have been generated in ${OUTPUT_DIRECTORY}/sphinx"
