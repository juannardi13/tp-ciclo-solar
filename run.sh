#!/bin/bash
# Corre todo el pipeline en orden. Uso: ./run.sh
set -e
cd "$(dirname "$0")"
for s in src/0[0-8]_*.py; do echo "== $s"; ./venv/bin/python "$s"; done
echo "== listo: ver out/"
