@echo off
rem Run the 745 automatic tests.
title ai_rf2 - 7_RUN_TESTS
call "%~dp0..\demo_windows.bat" 7 %1
set RC=%errorlevel%
echo.
echo ----------------------------------------------------------
if "%RC%"=="0" (echo Finished.) else (echo Finished with errors - exit code %RC%)
echo Press any key to close this window.
pause >nul
