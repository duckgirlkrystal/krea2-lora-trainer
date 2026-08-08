@echo off
setlocal
cd /d "%~dp0"
title Krystal's LoRA Trainer - Emergency Shutdown

echo.
echo  ==============================================================
echo   EMERGENCY SHUTDOWN
echo  ==============================================================
echo.
echo  This shuts down every computer this trainer rented, so you
echo  stop being charged for them.
echo.
echo  If anything else is running on your RunPod account, it will
echo  list those separately and ask before touching them.
echo.
echo  It is always safe to run this, even if nothing is running.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "trainer\bootstrap_windows.ps1" -Mode cloud
if errorlevel 1 goto failed

"trainer\.venv\Scripts\python.exe" "trainer\train_runpod.py" --shutdown-all
if errorlevel 1 goto failed

echo.
echo  --------------------------------------------------------------
echo   All clear.
echo.
echo   To be completely certain, you can also open this page
echo   in your browser and check that the list is empty:
echo       https://console.runpod.io/pods
echo  --------------------------------------------------------------
echo.
pause
exit /b 0

:failed
echo.
echo  --------------------------------------------------------------
echo   Could not shut things down automatically.
echo.
echo   PLEASE DO THIS BY HAND RIGHT NOW:
echo     1. Open  https://console.runpod.io/pods
echo     2. Delete anything in the list.
echo  --------------------------------------------------------------
echo.
pause
exit /b 1
