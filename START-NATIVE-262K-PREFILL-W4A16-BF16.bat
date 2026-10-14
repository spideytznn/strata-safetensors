@echo off
title strata-safetensors - W4A16 BF16 expert prefill - single request - MTP - 8880
echo Trial: NVFP4 weights temporarily expanded to BF16; BF16 activations and Tensor Core GEMM. Prefill 8192; BF16 KV; native MTP; port 8880.
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_safetensors.ps1" -Config "%~dp0config\native\rtx5090-262k-prefill-tc-bf16-w4a16.json" -Port 8880
if errorlevel 1 pause
