@echo off
title strata-safetensors - experimental two-slot MTP - 8880
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_safetensors.ps1" -Config "%~dp0config\native\rtx5090-262k-parallel-mtp.json" -Port 8880
if errorlevel 1 pause
