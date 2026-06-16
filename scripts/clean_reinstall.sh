#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

echo "==> Clearing __pycache__ directories..."
find . -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
echo "    Done."

echo "==> Reinstalling package in editable mode..."
pip install -e .
echo "    Done."

echo ""
echo "Package is now installed from source. Any code changes take effect immediately."
