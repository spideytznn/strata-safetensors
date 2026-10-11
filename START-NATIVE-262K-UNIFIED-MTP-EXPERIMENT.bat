@echo off
title strata-safetensors - experimental CPU/GPU batching with dual MTP - 8880
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_safetensors.ps1" -Config "%~dp0config\native\rtx5090-262k-unified-mtp-experiment.json" -Port 8880
if errorlevel 1 pause
