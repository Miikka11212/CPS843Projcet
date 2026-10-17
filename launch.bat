@echo off
setlocal
cd /d "%~dp0"
set "SIGNLAB_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%SIGNLAB_PYTHON%" goto run
if exist "%~dp0runtime_path.txt" set /p SIGNLAB_PYTHON=<"%~dp0runtime_path.txt"
if exist "%SIGNLAB_PYTHON%" goto run
echo Creating the Python environment. This is only needed on first launch.
powershell -NoProfile -ExecutionPolicy Bypass -File "%~dp0setup.ps1"
if errorlevel 1 goto failed
set "SIGNLAB_PYTHON=%~dp0.venv\Scripts\python.exe"
:run
"%SIGNLAB_PYTHON%" start_demo.py
if errorlevel 1 goto failed
exit /b 0
:failed
echo Launch failed. See the message above or demo.log in this folder.
pause
exit /b 1
