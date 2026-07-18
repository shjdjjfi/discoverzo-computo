#!/usr/bin/env bash
set -euo pipefail
quarto add computorg/computo-quarto-extension@v1.0.4 --no-prompt
quarto render
