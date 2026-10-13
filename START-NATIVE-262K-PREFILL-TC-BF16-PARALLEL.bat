@echo off
title strata-safetensors - BF16 KV Tensor Core - 2 concurrent requests - MTP - 8880
echo FP4 Tensor Core prefill 8192; BF16 KV Tensor Core; original expert decode. Native MTP and batch MTP enabled; spec 3. Two request slots.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_safetensors.ps1" -Config "%~dp0config\native\rtx5090-262k-prefill-tc-bf16-parallel.json" -Port 8880
if errorlevel 1 pause
