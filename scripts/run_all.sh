#!/bin/bash
set -e  # Exit on error

# Ensure script is run from repo root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the repository root."
    exit 1
fi

echo "=========================================="
echo "Sumerian NMT - Full Pipeline"
echo "=========================================="

# Step 0: Setup (skip if already done)
if [ ! -d "venv" ]; then
    ./scripts/00_setup.sh
fi
source venv/bin/activate

# Run pipeline steps
./scripts/01_extract_corpus.sh
./scripts/03_augment.sh

echo ""
echo "=========================================="
echo "Pipeline complete!"
echo "=========================================="
