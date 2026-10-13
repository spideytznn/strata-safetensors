@echo off
title strata-safetensors - prefill/decode overlap - 8880
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_safetensors.ps1" -Config "%~dp0config\native\rtx5090-262k-prefill-overlap.json" -Port 8880
if errorlevel 1 pause
