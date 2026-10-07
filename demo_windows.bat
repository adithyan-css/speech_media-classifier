@echo off
rem ======================================================================
rem  ai_rf2 - one-click launcher for Windows (double-click this file)
rem  Optional: "demo_windows.bat 5" runs menu item 5 directly
rem  ("demo_windows.bat P" opens a file window to choose a song).
rem ======================================================================
setlocal EnableExtensions EnableDelayedExpansion
chcp 65001 >nul
title ai_rf2 - speech / music classifier
cd /d "%~dp0"
set "HERE=%~dp0"
set "PYTHONPATH=%HERE%src"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONDONTWRITEBYTECODE=1"
if not defined AI_RF2_DATA set "AI_RF2_DATA=%LOCALAPPDATA%\ai_rf2_data"
if not exist "%AI_RF2_DATA%" mkdir "%AI_RF2_DATA%"
set "BACKEND=torch"

rem ---------- 1. find a Python that already has PyTorch ----------
set "PY="
if defined AI_RF2_PYTHON set "PY=%AI_RF2_PYTHON%"
if not defined PY if exist "%HERE%..\AAC_RF2\.venv\Scripts\python.exe" (
    "%HERE%..\AAC_RF2\.venv\Scripts\python.exe" -c "import torch" >nul 2>&1 && set PY="%HERE%..\AAC_RF2\.venv\Scripts\python.exe"
)
if not defined PY (
    py -3 -c "import torch" >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
    python -c "import torch" >nul 2>&1 && set "PY=python"
)
if not defined PY (
    py -3 --version >nul 2>&1 && set "PY=py -3"
)
if not defined PY (
    python --version >nul 2>&1 && set "PY=python"
)
if not defined PY (
    echo Python 3.10 or newer was not found.
    echo Install it from https://www.python.org ^(tick "Add python.exe to PATH"^) and run this again.
    pause
    exit /b 1
)

rem ---------- 2. make sure every package is there (only installs what is missing) ----------
set "MARK=%AI_RF2_DATA%\setup_ok.txt"
set "LASTPY="
if exist "%MARK%" set /p LASTPY=<"%MARK%"
if not "!LASTPY!"=="!PY!" (
    echo Checking the Python environment - first run only ...
    %PY% "%HERE%scripts\check_env.py" --install
    if errorlevel 1 (
        echo.
        echo Setup did not finish - see the messages above.
        pause
        exit /b 1
    )
    >"%MARK%" echo !PY!
)

set "CHOICE=%~1"
set "ONESHOT=%~1"

:menu
if not "!CHOICE!"=="" goto dispatch
cls
echo ======================================================================
echo   ai_rf2  -  AI Radio Announcement Classifier  (speech / music demo)
echo ======================================================================
echo   Python : !PY!
echo   Model  : !BACKEND!    (torch = PyTorch model, onnx = 8-bit ODROID model)
echo.
echo   1  LIVE demo   - radio programme 1  (sound + live window)
echo   2  LIVE demo   - radio programme 2
echo   3  LIVE demo   - your own audio file (mp3 / wav / flac)
echo   P  LIVE demo   - CHOOSE a song / audio file on this laptop (plays + classifies)
echo   4  LIVE demo   - microphone or line-in
echo   5  REPORT demo - both programmes: result table + timeline pictures
echo   6  Classify one audio file (text only)
echo   7  Run the automatic tests
echo   8  Check the Python setup
echo   9  Retrain everything from scratch (downloads 3.2 GB, needs ~12 GB disk)
echo   T  Full self-test of this machine (about 5 minutes, silent)
echo   B  Switch model  PyTorch ^<-^> 8-bit ONNX
echo   0  Exit
echo.
set /p "CHOICE=Type a number and press Enter: "

:dispatch
set "RC=0"
if /i "!CHOICE!"=="0" exit /b 0
if /i "!CHOICE!"=="B" (
    if "!BACKEND!"=="torch" (set "BACKEND=onnx") else (set "BACKEND=torch")
    set "CHOICE="
    goto menu
)
if "!CHOICE!"=="1" %PY% -m ai_rf2.live --programme 0 --backend !BACKEND! %AI_RF2_LIVE_ARGS%
if "!CHOICE!"=="2" %PY% -m ai_rf2.live --programme 1 --backend !BACKEND! %AI_RF2_LIVE_ARGS%
if "!CHOICE!"=="3" (
    set "F=%~2"
    if "!F!"=="" set /p "F=Drag an audio file into this window and press Enter: "
    set "F=!F:"=!"
    %PY% -m ai_rf2.live --file "!F!" --backend !BACKEND! %AI_RF2_LIVE_ARGS%
)
if /i "!CHOICE!"=="P" %PY% -m ai_rf2.live --pick --backend !BACKEND! %AI_RF2_LIVE_ARGS%
if "!CHOICE!"=="4" %PY% -m ai_rf2.live --mic --backend !BACKEND! %AI_RF2_LIVE_ARGS%
if "!CHOICE!"=="5" (
    %PY% -m ai_rf2.demo --backend !BACKEND! --compare
    set "RC=!errorlevel!"
    if "!ONESHOT!"=="" start "" "%HERE%reports\demo"
)
if "!CHOICE!"=="6" (
    set "F=%~2"
    if "!F!"=="" set /p "F=Drag an audio file into this window and press Enter: "
    set "F=!F:"=!"
    %PY% -m ai_rf2.infer "!F!" --backend !BACKEND!
)
if "!CHOICE!"=="7" %PY% -m pytest "%HERE%tests" -q -m "not slow" -p no:cacheprovider --junitxml="%HERE%reports\tests_windows.xml"
if /i "!CHOICE!"=="T" %PY% "%HERE%scripts\selftest.py" %AI_RF2_SELFTEST_ARGS%
if "!CHOICE!"=="8" %PY% "%HERE%scripts\check_env.py" --install
if "!CHOICE!"=="9" (
    echo Big data goes to %AI_RF2_DATA%  ^(outside OneDrive^)
    %PY% "%HERE%scripts\run_all.py"
)
if "!RC!"=="0" set "RC=!errorlevel!"
if not "!ONESHOT!"=="" exit /b !RC!
echo.
pause
set "CHOICE="
goto menu
