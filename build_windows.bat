@echo off
echo =============================================
echo  Shorts Auto Editor - Windows Build Script
echo =============================================
echo.

echo [1/3] Installing dependencies...
pip install -r requirements.txt
pip install -r requirements_gui.txt
pip install pyinstaller
echo.

echo [2/3] Checking ffmpeg...
where ffmpeg >nul 2>&1
if %errorlevel% neq 0 (
    echo WARNING: ffmpeg not found in PATH.
    echo Please download ffmpeg from https://ffmpeg.org/download.html
    echo and place ffmpeg.exe in this folder or add it to your PATH.
    echo.
)

echo [3/3] Building executable...
pyinstaller ^
  --onedir ^
  --windowed ^
  --name "ShortsAutoEditor" ^
  --collect-all open_clip ^
  --collect-all timm ^
  --hidden-import=open_clip ^
  --hidden-import=open_clip.pretrained ^
  --hidden-import=timm ^
  --hidden-import=tkinterdnd2 ^
  gui.py

echo.
echo =============================================
echo  Done! Output: dist\ShortsAutoEditor\
echo  Send the entire ShortsAutoEditor folder.
echo =============================================
pause
