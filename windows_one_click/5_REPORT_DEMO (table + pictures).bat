@echo off
rem REPORT demo: segment-by-segment table and timeline pictures for both programmes.
title ai_rf2 - 5_REPORT_DEMO (table + pictures)
call "%~dp0..\demo_windows.bat" 5 %1
set RC=%errorlevel%
start "" "%~dp0..\reports\demo"
echo.
echo ----------------------------------------------------------
if "%RC%"=="0" (echo Finished.) else (echo Finished with errors - exit code %RC%)
echo Press any key to close this window.
pause >nul
