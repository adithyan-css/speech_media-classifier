@echo off
rem LIVE demo from the microphone / line-in. Hold a phone playing radio near the laptop.
title ai_rf2 - 4_LIVE_DEMO microphone or line-in
call "%~dp0..\demo_windows.bat" 4 %1
set RC=%errorlevel%
echo.
echo ----------------------------------------------------------
if "%RC%"=="0" (echo Finished.) else (echo Finished with errors - exit code %RC%)
echo Press any key to close this window.
pause >nul
