@echo off
rem Full self-test: every part of the project, nothing is played aloud.
title ai_rf2 - 0_SELF_TEST (check this laptop, silent, 5 min)
call "%~dp0..\demo_windows.bat" T %1
set RC=%errorlevel%
echo.
echo ----------------------------------------------------------
if "%RC%"=="0" (echo Finished.) else (echo Finished with errors - exit code %RC%)
echo Press any key to close this window.
pause >nul
