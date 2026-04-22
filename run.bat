@echo off
cd /d "%~dp0"
set "PATH=%~dp0;%PATH%"
python gui.py
