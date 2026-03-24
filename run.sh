#!/usr/bin/env bash
set -e

echo "======================================================="
echo "Football Performance and Team Success - Dashboard Setup"
echo "======================================================="
echo

echo "[1/2] Installing Dependencies..."
pip install -r requirements.txt

echo
echo "[2/2] Starting Streamlit Dashboard..."
echo
streamlit run dashboard.py
