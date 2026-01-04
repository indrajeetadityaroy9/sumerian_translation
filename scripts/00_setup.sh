#!/bin/bash
set -e  # Exit on error

# Ensure script is run from repo root
if [ ! -f "pyproject.toml" ]; then
    echo "Error: Please run this script from the repository root."
    exit 1
fi

echo "=========================================="
echo "Sumerian NMT - Environment Setup"
echo "=========================================="

# Create virtual environment
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Install package in editable mode
echo "Installing package..."
pip install -e .

# Install dev dependencies
echo "Installing dev dependencies..."
pip install -e ".[dev]"

echo ""
echo "Setup complete!"
echo "Activate with: source venv/bin/activate"
