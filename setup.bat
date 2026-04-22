@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul
title Shorts Auto Editor - 설치 중...

set "DIR=%~dp0"

echo.
echo  ╔════════════════════════════════════════╗
echo  ║      Shorts Auto Editor  설치         ║
echo  ╚════════════════════════════════════════╝
echo.

REM ── [1/5] Python 확인 / 설치 ─────────────────────────────────────────────
echo [1/5] Python 확인 중...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo     Python이 없습니다. 자동 설치 중... ^(잠시 기다려주세요^)
    curl -L "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" ^
         -o "%TEMP%\py_setup.exe" --silent --show-error
    "%TEMP%\py_setup.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0
    del "%TEMP%\py_setup.exe" >nul 2>&1

    REM PATH 새로고침 후 재확인
    set "PYPATH=%LOCALAPPDATA%\Programs\Python\Python311"
    set "PATH=!PYPATH!;!PYPATH!\Scripts;%PATH%"

    python --version >nul 2>&1
    if !errorlevel! neq 0 (
        echo.
        echo  ❌ Python 설치 실패.
        echo     https://www.python.org 에서 직접 설치 후 다시 실행해주세요.
        pause & exit /b 1
    )
    echo     Python 설치 완료
) else (
    for /f "tokens=*" %%v in ('python --version 2^>^&1') do echo     %%v 확인
)

REM ── [2/5] GPU 확인 ────────────────────────────────────────────────────────
echo.
echo [2/5] GPU 확인 중...
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    for /f "tokens=*" %%g in ('nvidia-smi --query-gpu=name --format=csv^,noheader 2^>nul') do (
        echo     NVIDIA GPU 감지: %%g
    )
    set "TORCH_URL=https://download.pytorch.org/whl/cu121"
    set "GPU_MSG=CUDA"
) else (
    echo     GPU 없음 ^(CPU 모드로 실행됩니다^)
    set "TORCH_URL=https://download.pytorch.org/whl/cpu"
    set "GPU_MSG=CPU"
)

REM ── [3/5] Python 패키지 설치 ──────────────────────────────────────────────
echo.
echo [3/5] Python 패키지 설치 중... ^(5~15분 소요^)
python -m pip install --upgrade pip -q
python -m pip install torch torchvision --index-url !TORCH_URL! -q
if %errorlevel% neq 0 (
    echo  ❌ PyTorch 설치 실패. 인터넷 연결을 확인해주세요.
    pause & exit /b 1
)
python -m pip install -r "%DIR%requirements.txt" -q
python -m pip install -r "%DIR%requirements_gui.txt" -q
echo     패키지 설치 완료 ^(!GPU_MSG! 버전^)

REM ── [4/5] AI 모델 다운로드 ────────────────────────────────────────────────
echo.
echo [4/5] AI 모델 다운로드 중... ^(약 350MB, 최초 1회만^)
python -c "import open_clip; open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')"
if %errorlevel% neq 0 (
    echo  ❌ 모델 다운로드 실패. 인터넷 연결을 확인해주세요.
    pause & exit /b 1
)
echo     모델 다운로드 완료

REM ── [5/5] ffmpeg 설치 ─────────────────────────────────────────────────────
echo.
echo [5/5] ffmpeg 확인 중...
if exist "%DIR%ffmpeg.exe" (
    echo     ffmpeg 이미 있음
) else (
    where ffmpeg >nul 2>&1
    if !errorlevel! equ 0 (
        echo     ffmpeg 이미 설치됨 ^(시스템^)
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
        echo     ffmpeg 설치 완료
    )
)

REM ── 바로가기 생성 ─────────────────────────────────────────────────────────
echo.
echo  바탕화면 바로가기 생성 중...

REM run.vbs 생성 (콘솔 창 없이 실행)
(
echo Set fso = CreateObject^("Scripting.FileSystemObject"^)
echo Set wsh = CreateObject^("WScript.Shell"^)
echo dir = fso.GetParentFolderName^(WScript.ScriptFullName^)
echo wsh.CurrentDirectory = dir
echo wsh.Environment^("Process"^)^("PATH"^) = dir ^& ";" ^& wsh.Environment^("Process"^)^("PATH"^)
echo wsh.Run "python " ^& Chr^(34^) ^& dir ^& "\gui.py" ^& Chr^(34^), 0, False
) > "%DIR%run.vbs"

powershell -Command ^
  "$ws = New-Object -ComObject WScript.Shell; ^
   $s  = $ws.CreateShortcut([System.IO.Path]::Combine($env:USERPROFILE, 'Desktop', 'Shorts Auto Editor.lnk')); ^
   $s.TargetPath     = 'wscript.exe'; ^
   $s.Arguments      = '\"%DIR%run.vbs\"'; ^
   $s.WorkingDirectory = '%DIR%'; ^
   $s.Description    = 'Shorts Auto Editor'; ^
   $s.Save()"

echo.
echo  ╔════════════════════════════════════════╗
echo  ║   ✅ 설치 완료!                        ║
echo  ║                                        ║
echo  ║   바탕화면의                           ║
echo  ║   'Shorts Auto Editor' 를             ║
echo  ║   더블클릭해서 실행하세요 🎬           ║
echo  ╚════════════════════════════════════════╝
echo.
pause
