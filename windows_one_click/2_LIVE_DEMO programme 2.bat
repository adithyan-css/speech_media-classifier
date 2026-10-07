@echo off
rem LIVE demo: plays radio programme 2 and shows the live decision window.
title ai_rf2 - 2_LIVE_DEMO programme 2
call "%~dp0..\demo_windows.bat" 2 %1
set RC=%errorlevel%
echo.
echo ----------------------------------------------------------
if "%RC%"=="0" (echo Finished.) else (echo Finished with errors - exit code %RC%)
echo Press any key to close this window.
pause >nul
