@echo off
setlocal
title Strata Fusion 0.1.41 UD-Q6_K_XL No MTP - port 8880
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "G:\Strata\Strata\Start-Strata-Fusion-Q6KL-NoMTP.ps1" %*
set "STRATA_EXIT=%ERRORLEVEL%"
if /I "%~1"=="-CheckOnly" exit /b %STRATA_EXIT%
echo.
echo [Strata Fusion Q6KL NoMTP] Server stopped. Press any key to close this window.
pause >nul
exit /b %STRATA_EXIT%
