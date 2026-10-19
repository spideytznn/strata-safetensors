@echo off
title Strata NVFP4 - 2 concurrent requests - 8880
echo Original NVIDIA NVFP4 safetensors with the Strata v0.2.0-beta engine.
echo Two request slots, native batch MTP, BF16 KV, W4A4 prefill, GPU vision.
echo Web UI: http://127.0.0.1:8880
echo API:    http://127.0.0.1:8880/v1
echo Keep this window open. Press Ctrl+C to stop.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_safetensors.ps1" -Config "%~dp0strata-nvfp4-native.json" -Port 8880
if errorlevel 1 pause
