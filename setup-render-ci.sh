#!/usr/bin/env bash
set -euo pipefail

echo "Checking Python and Jupyter environment..."
python --version
python -m jupyter --version
quarto check jupyter
