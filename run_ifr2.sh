#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -x "./venv/bin/streamlit" ]]; then
  echo "Erro: venv não encontrado em ./venv"
  echo "Crie o ambiente com: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
  exit 1
fi

exec ./venv/bin/streamlit run ifr2_app.py
