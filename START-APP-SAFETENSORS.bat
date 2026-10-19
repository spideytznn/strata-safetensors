@echo off
title Strata Safetensors - v0.2.0-beta
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0tools\run_safetensors_app.ps1"
if errorlevel 1 pause
