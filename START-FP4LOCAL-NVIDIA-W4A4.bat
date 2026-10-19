@echo off
title Strata FP4-Local - EXPERIMENTAL NVIDIA W4A4 - 8880
echo EXPERIMENTAL: FP4 Tensor Core expert decode; 2 concurrent requests; prefill 8192.
echo CPU AVX512 W4A4 for cold experts; GPU FP4 Tensor Cores for hot experts. Experimental.
echo Experimental: greedy output can change when experts move between CPU and GPU.
echo Stop the stable server before starting this one.
echo Web UI: http://127.0.0.1:8880
echo API:    http://127.0.0.1:8880/v1
echo Keep this window open. Press Ctrl+C to stop.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_safetensors.ps1" -Config "%~dp0config\native\rtx5090-fp4local-nvidia-w4a4.json" -Port 8880
if errorlevel 1 pause
