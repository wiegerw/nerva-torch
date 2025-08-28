#!/bin/bash
set -euo pipefail

# Resolve repository root and script directory (so it works from any CWD)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Output directory at the repository root (so CI publishes ./output_directory)
OUTPUT_DIRECTORY="${REPO_ROOT}/output_directory"
mkdir -p "${OUTPUT_DIRECTORY}"

# Pre-clean only our outputs (do not touch other artifacts like PDFs)
rm -f "${OUTPUT_DIRECTORY}/nerva-torch.html"

# Common Asciidoctor options (no bibtex-file here)
COMMON_OPTS=(-r asciidoctor-bibtex -a "source-highlighter=rouge" -D "${OUTPUT_DIRECTORY}")

# Build Python manual
pushd "${SCRIPT_DIR}/asciidoc" >/dev/null
asciidoctor "${COMMON_OPTS[@]}" -a "bibtex-file=../latex/nerva.bib" nerva-torch.adoc
echo "The HTML page has been generated in ${OUTPUT_DIRECTORY}/nerva-torch.html"
popd >/dev/null
