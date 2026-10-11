@echo off
title strata-safetensors - experimental ragged prefill - 4 slots - 8880
echo Experimental ragged prefill changes floating-point reduction order. See docs\NATIVE_BATCH_SCHEDULER.md.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_safetensors.ps1" -Config "%~dp0config\native\rtx5090-262k-scheduler-ragged.json" -Port 8880
if errorlevel 1 pause
