@echo off
setlocal enabledelayedexpansion
title ClipTrace - 설치

set "DIR=%~dp0"

echo.
echo ============================================
echo   ClipTrace - 설치
echo ============================================
echo.

REM [1/4] Python
echo [1/4] Python 확인 중...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo     Python이 없습니다. Python 3.11 다운로드 중...
    curl -L "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" ^
         -o "%TEMP%\py_setup.exe" --silent --show-error
    "%TEMP%\py_setup.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    del "%TEMP%\py_setup.exe" >nul 2>&1
    set "PYPATH=%LOCALAPPDATA%\Programs\Python\Python311"
    set "PATH=!PYPATH!;!PYPATH!\Scripts;%PATH%"
    python --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo.
        echo 오류: Python 설치 실패.
        echo https://www.python.org 에서 Python 3.11을 직접 설치해주세요.
        pause & exit /b 1
    )
    echo     Python 설치 완료.
) else (
    for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo     %%v 확인됨.
)

REM [2/4] GPU check
echo.
echo [2/4] GPU 확인 중...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo     NVIDIA GPU 감지됨 - CUDA 버전 설치 예정
    set "TORCH_URL=https://download.pytorch.org/whl/cu121"
) else (
    echo     GPU 없음 - CPU 버전 설치 예정
    set "TORCH_URL=https://download.pytorch.org/whl/cpu"
)

REM [3/4] Python packages
echo.
echo [3/4] Python 패키지 설치 중 (10~20분 소요)...
python -m pip install --upgrade pip -q
python -m pip install torch torchvision --index-url !TORCH_URL! -q
if %errorlevel% neq 0 (
    echo 오류: PyTorch 설치 실패. 인터넷 연결을 확인해주세요.
    pause & exit /b 1
)
python -m pip install -r "%DIR%requirements.txt" -q
python -m pip install -r "%DIR%requirements_gui.txt" -q
echo     패키지 설치 완료.

REM [4/4] ffmpeg
echo.
echo [4/4] ffmpeg 확인 중...
if exist "%DIR%ffmpeg.exe" (
    echo     ffmpeg 이미 있음.
) else (
    where ffmpeg >nul 2>&1
    if !errorlevel! equ 0 (
        echo     시스템 PATH에서 ffmpeg 발견.
    ) else (
        echo     ffmpeg 다운로드 중...
        curl -L "https://www.gyan.dev/ffmpeg/builds/ffmpeg-release-essentials.zip" ^
             -o "%TEMP%\ffmpeg.zip" --progress-bar
        powershell -command "Expand-Archive -Path '%TEMP%\ffmpeg.zip' -DestinationPath '%TEMP%\ffmpeg_ex' -Force"
        for /d %%i in ("%TEMP%\ffmpeg_ex\ffmpeg-*") do (
            copy "%%i\bin\ffmpeg.exe"  "%DIR%ffmpeg.exe"  >nul
            copy "%%i\bin\ffprobe.exe" "%DIR%ffprobe.exe" >nul
        )
        rmdir /s /q "%TEMP%\ffmpeg_ex" >nul 2>&1
        del "%TEMP%\ffmpeg.zip" >nul 2>&1
        echo     ffmpeg 설치 완료.
    )
)

REM Create shortcuts
echo.
echo 바로가기 생성 중...
powershell -Command ^
    "$pythonw = (Get-Command python.exe -ErrorAction SilentlyContinue).Source -replace 'python.exe','pythonw.exe';" ^
    "$ws = New-Object -ComObject WScript.Shell;" ^
    "foreach ($dest in @([IO.Path]::Combine($env:USERPROFILE,'Desktop','ClipTrace.lnk'), '%DIR%ClipTrace.lnk')) {" ^
    "  $s = $ws.CreateShortcut($dest);" ^
    "  $s.TargetPath = $pythonw;" ^
    "  $s.Arguments = ('\"' + '%DIR%gui.py' + '\"');" ^
    "  $s.WorkingDirectory = '%DIR%';" ^
    "  $s.Description = 'ClipTrace';" ^
    "  $s.Save()" ^
    "}"

echo.
echo ============================================
echo   설치 완료!
echo   바탕화면의 "ClipTrace" 바로가기를
echo   더블클릭해서 실행하세요.
echo.
echo   참고: AI 모델 (약 85MB)은 첫 실행 시
echo   자동으로 다운로드됩니다.
echo ============================================
timeout /t 3 /nobreak >nul
