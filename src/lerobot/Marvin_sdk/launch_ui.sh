#!/bin/bash
# Quick launcher for Robot Arm Debug UI

cd "$(dirname "$0")"

echo "=========================================="
echo "  Robot Arm Debug UI Launcher"
echo "=========================================="
echo ""
echo "Starting English UI..."
echo ""

python3 arm_ui_en.py

echo ""
echo "UI closed."
