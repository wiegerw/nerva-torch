#!/bin/bash
set -e

# Resolve repository root and script directory (so it works from any CWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Output directory at the repository root (so CI publishes ./output_directory)
OUTPUT_DIRECTORY="${REPO_ROOT}/output_directory"
mkdir -p "${OUTPUT_DIRECTORY}"

# Ensure Sphinx + theme are available
python3 -m pip install --upgrade sphinx sphinx-rtd-theme

# Build HTML docs from ../docs_sphinx into output_directory
sphinx-build -b html sphinx $OUTPUT_DIRECTORY/sphinx

# Copy landing page
cp index.html $OUTPUT_DIRECTORY
