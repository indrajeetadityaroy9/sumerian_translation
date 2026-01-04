#!/bin/bash
set -e  # Exit on error

# Ensure script is run from repo root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the repository root."
    exit 1
fi

echo "=========================================="
echo "Step 1: Extract ETCSL Corpus"
echo "=========================================="

sumerian-extract --output-dir output

echo "Extraction complete! Output in output/"
