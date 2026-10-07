@echo off
rem LIVE demo on any song or audio file from this laptop.
rem A normal "Open file" window appears: pick an mp3 / wav / flac / m4a ... The song plays
rem through the speakers while the live window shows speech / music and the virtual TA flag.
rem In the live window, "Choose another file ..." switches to the next song.
rem (Dragging a file onto this shortcut also works.)
title ai_rf2 - 11_LIVE_DEMO choose audio from this laptop
if not "%~1"=="" goto dropped
call "%~dp0..\demo_windows.bat" P
goto done
:dropped
call "%~dp0..\demo_windows.bat" 3 %1
:done
set RC=%errorlevel%
echo.
echo ----------------------------------------------------------
if "%RC%"=="0" (echo Finished.) else (echo Finished with errors - exit code %RC%)
echo Press any key to close this window.
pause >nul
