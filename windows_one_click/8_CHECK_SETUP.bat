@echo off
rem Check / repair the Python packages (never touches PyTorch).
title ai_rf2 - 8_CHECK_SETUP
call "%~dp0..\demo_windows.bat" 8 %1
set RC=%errorlevel%
echo.
echo ----------------------------------------------------------
if "%RC%"=="0" (echo Finished.) else (echo Finished with errors - exit code %RC%)
echo Press any key to close this window.
pause >nul
