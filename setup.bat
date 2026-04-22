@echo off
setlocal enabledelayedexpansion
title Shorts Auto Editor - Setup

set "DIR=%~dp0"

echo.
echo ============================================
echo   Shorts Auto Editor - Setup
echo ============================================
echo.

REM [1/5] Python
echo [1/5] Checking Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo     Python not found. Downloading Python 3.11...
    curl -L "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" ^
         -o "%TEMP%\py_setup.exe" --silent --show-error
    "%TEMP%\py_setup.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    del "%TEMP%\py_setup.exe" >nul 2>&1
    set "PYPATH=%LOCALAPPDATA%\Programs\Python\Python311"
    set "PATH=!PYPATH!;!PYPATH!\Scripts;%PATH%"
    python --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo.
        echo ERROR: Python install failed.
        echo Please install Python 3.11 from https://www.python.org manually.
        pause & exit /b 1
    )
    echo     Python installed.
) else (
    for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo     %%v found.
)

REM [2/5] GPU check
echo.
echo [2/5] Checking GPU...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo     NVIDIA GPU detected - will install CUDA version
    set "TORCH_URL=https://download.pytorch.org/whl/cu121"
) else (
    echo     No GPU found - will install CPU version
    set "TORCH_URL=https://download.pytorch.org/whl/cpu"
)

REM [3/5] Python packages
echo.
echo [3/5] Installing Python packages (10-15 min)...
python -m pip install --upgrade pip -q
python -m pip install torch torchvision --index-url !TORCH_URL! -q
if %errorlevel% neq 0 (
    echo ERROR: PyTorch install failed. Check internet connection.
    pause & exit /b 1
)
python -m pip install -r "%DIR%requirements.txt" -q
python -m pip install -r "%DIR%requirements_gui.txt" -q
echo     Packages installed.

REM [4/5] CLIP model
echo.
echo [4/5] Downloading CLIP model (~350MB, first time only)...
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')"
if %errorlevel% neq 0 (
    echo ERROR: Model download failed. Check internet connection.
    pause & exit /b 1
)
echo     Model ready.

REM [5/5] ffmpeg
echo.
echo [5/5] Checking ffmpeg...
if exist "%DIR%ffmpeg.exe" (
    echo     ffmpeg already present.
) else (
    where ffmpeg >nul 2>&1
    if !errorlevel! equ 0 (
        echo     ffmpeg found in system PATH.
    ) else (
        echo     Downloading ffmpeg...
        curl -L "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" ^
             -o "%TEMP%\ffmpeg.zip" --progress-bar
        powershell -command "Expand-Archive -Path '%TEMP%\ffmpeg.zip' -DestinationPath '%TEMP%\ffmpeg_ex' -Force"
        for /d %%i in ("%TEMP%\ffmpeg_ex\ffmpeg-*") do (
            copy "%%i\bin\ffmpeg.exe"  "%DIR%ffmpeg.exe"  >nul
            copy "%%i\bin\ffprobe.exe" "%DIR%ffprobe.exe" >nul
        )
        rmdir /s /q "%TEMP%\ffmpeg_ex" >nul 2>&1
        del "%TEMP%\ffmpeg.zip" >nul 2>&1
        echo     ffmpeg installed.
    )
)

REM Create desktop shortcut using pythonw (no console window)
echo.
echo Creating desktop shortcut...
powershell -Command ^
    "$pythonw = (Get-Command python.exe -ErrorAction SilentlyContinue).Source -replace 'python.exe','pythonw.exe';" ^
    "$ws = New-Object -ComObject WScript.Shell;" ^
    "$s = $ws.CreateShortcut([IO.Path]::Combine($env:USERPROFILE,'Desktop','Shorts Auto Editor.lnk'));" ^
    "$s.TargetPath = $pythonw;" ^
    "$s.Arguments = ('\"' + '%DIR%gui.py' + '\"');" ^
    "$s.WorkingDirectory = '%DIR%';" ^
    "$s.Description = 'Shorts Auto Editor';" ^
    "$s.Save()"

echo.
echo ============================================
echo   Setup complete!
echo   Double-click "Shorts Auto Editor"
echo   on your Desktop to start.
echo ============================================
echo.
pause
