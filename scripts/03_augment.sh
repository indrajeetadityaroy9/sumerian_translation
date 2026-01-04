#!/bin/bash
set -e  # Exit on error

# Ensure script is run from repo root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the repository root."
    exit 1
fi

echo "=========================================="
echo "Step 3: Graph-Based Entity Substitution"
echo "=========================================="

sumerian-augment --parallel --output-format both

echo "Augmentation complete!"
