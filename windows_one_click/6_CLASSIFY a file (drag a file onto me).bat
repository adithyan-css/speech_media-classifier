@echo off
rem Classify one audio file and print when speech starts and stops.
title ai_rf2 - 6_CLASSIFY a file (drag a file onto me)
call "%~dp0..\demo_windows.bat" 6 %1
set RC=%errorlevel%
echo.
echo ----------------------------------------------------------
if "%RC%"=="0" (echo Finished.) else (echo Finished with errors - exit code %RC%)
echo Press any key to close this window.
pause >nul
