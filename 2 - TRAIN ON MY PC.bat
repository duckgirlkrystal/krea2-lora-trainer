@echo off
setlocal
cd /d "%~dp0"
title Krystal's LoRA Trainer - This PC

echo.
echo  ==============================================================
echo   KRYSTAL'S LORA TRAINER  -  YOUR OWN GRAPHICS CARD
echo  ==============================================================
echo.
echo  First it checks whether your PC can handle this.
echo  If it can't, it will say so plainly and stop. Nothing breaks.
echo.
echo  The very first run downloads about 35 GB and can take
echo  well over an hour before training even starts. That is
echo  normal. Later runs skip all of it.
echo.

powershell -NoProfile -ExecutionPolicy Bypass -File "trainer\bootstrap_windows.ps1" -Mode local
if errorlevel 1 goto failed

"trainer\.venv\Scripts\python.exe" "trainer\train_local.py"
if errorlevel 1 goto failed

goto finished

:failed
echo.
echo  --------------------------------------------------------------
echo   Something went wrong. The message above explains what.
echo.
echo   If your PC just isn't powerful enough, use the cloud
echo   instead - it works on any computer:
echo       1 - TRAIN IN THE CLOUD.bat
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
