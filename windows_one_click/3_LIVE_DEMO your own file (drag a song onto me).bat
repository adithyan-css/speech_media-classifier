@echo off
rem LIVE demo on your own audio file. Drag an mp3/wav/flac onto this file, or run it and paste a path.
title ai_rf2 - 3_LIVE_DEMO your own file (drag a song onto me)
call "%~dp0..\demo_windows.bat" 3 %1
set RC=%errorlevel%
echo.
echo ----------------------------------------------------------
if "%RC%"=="0" (echo Finished.) else (echo Finished with errors - exit code %RC%)
echo Press any key to close this window.
pause >nul
