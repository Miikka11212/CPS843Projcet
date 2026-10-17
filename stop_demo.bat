@echo off
setlocal
cd /d "%~dp0"
set "SIGNLAB_PYTHON=%~dp0.venv\Scripts\python.exe"
if exist "%SIGNLAB_PYTHON%" goto stop
if exist "%~dp0runtime_path.txt" set /p SIGNLAB_PYTHON=<"%~dp0runtime_path.txt"
if not exist "%SIGNLAB_PYTHON%" exit /b 0
:stop
"%SIGNLAB_PYTHON%" start_demo.py --stop
