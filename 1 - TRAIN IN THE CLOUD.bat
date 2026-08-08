@echo off
setlocal
cd /d "%~dp0"
title Krystal's LoRA Trainer - Cloud

echo.
echo  ==============================================================
echo   KRYSTAL'S LORA TRAINER  -  CLOUD
echo  ==============================================================
echo.
echo  Getting things ready. The first run takes a few minutes
echo  because it has to install some tools. After that it is quick.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "trainer\bootstrap_windows.ps1" -Mode cloud
if errorlevel 1 goto failed

"trainer\.venv\Scripts\python.exe" "trainer\train_runpod.py"
if errorlevel 1 goto failed

goto finished

:failed
echo.
echo  --------------------------------------------------------------
echo   Something went wrong. The message above explains what.
echo.
echo   If it mentions money or a rented computer still running,
echo   close this window and double-click:
echo       3 - EMERGENCY - SHUT DOWN CLOUD.bat
echo  --------------------------------------------------------------
echo.
pause
exit /b 1

:finished
echo.
echo  Done. Your results are in the "output" folder.
echo.
pause
exit /b 0
