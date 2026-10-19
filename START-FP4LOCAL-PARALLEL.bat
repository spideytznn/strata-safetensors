@echo off
title Strata FP4-Local - 2 concurrent requests - 8880
echo Strata FP4-Local: 2 concurrent requests, MTP OFF temporarily, auto GPU expert cache.
echo GPU vision: ON, BF16 encoder, image token cap 1024.
echo Web UI: http://127.0.0.1:8880
echo API:    http://127.0.0.1:8880/v1
echo Keep this window open. Press Ctrl+C to stop.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_safetensors.ps1" -Config "%~dp0strata-fp4local-native.json" -Port 8880
if errorlevel 1 pause
