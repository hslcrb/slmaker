#!/bin/bash
# run_cli.sh - slmaker Engine Executive (CLI) / CLI 실행 스크립트
# v1.0.0 Odyssey Full Release

# Activate Virtual Environment / 가상환경 활성화
source ./new_venv/bin/activate

# Execute CLI / CLI 실행 (Supports 'inference' argument)
python3 cli.py "$@"
