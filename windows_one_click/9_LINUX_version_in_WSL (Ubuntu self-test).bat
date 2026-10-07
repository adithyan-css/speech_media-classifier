@echo off
rem Runs the LINUX launcher inside WSL (Ubuntu) on this laptop - same project folder, Linux Python.
rem No argument = full Linux self-test (T). Otherwise the argument is a menu choice, e.g. 1.
title ai_rf2 - Linux (WSL)
where wsl >nul 2>&1 || (echo WSL is not installed on this PC. & pause & exit /b 1)
if "%~1"=="" (wsl --cd "%~dp0.." bash demo_linux.sh T) else (wsl --cd "%~dp0.." bash demo_linux.sh %1 %2)
echo.
echo ----------------------------------------------------------
echo Press any key to close this window.
pause >nul
