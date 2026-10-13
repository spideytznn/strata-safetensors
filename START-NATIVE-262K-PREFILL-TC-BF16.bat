@echo off
title strata-safetensors - BF16 KV Tensor Core - single request - MTP - 8880
echo FP4 Tensor Core prefill 8192; BF16 KV Tensor Core; original expert decode. Native MTP enabled; spec 3. One request at a time.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_safetensors.ps1" -Config "%~dp0config\native\rtx5090-262k-prefill-tc-bf16.json" -Port 8880
if errorlevel 1 pause
