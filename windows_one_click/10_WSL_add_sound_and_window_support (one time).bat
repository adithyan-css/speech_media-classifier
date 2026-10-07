@echo off
rem One-time: installs the two Ubuntu system libraries the LINUX live demo needs inside WSL:
rem   libportaudio2 (sound in/out) and python3-tk (the live window), plus the ALSA->PulseAudio
rem   bridge so sound reaches Windows through WSLg. Runs as root inside WSL (no password needed).
title ai_rf2 - WSL sound + window support
where wsl >nul 2>&1 || (echo WSL is not installed on this PC. & pause & exit /b 1)
wsl -u root -- bash -c "export DEBIAN_FRONTEND=noninteractive; apt-get update -qq && apt-get install -y -qq libportaudio2 python3-tk libasound2-plugins >/dev/null && echo packages installed; if [ -S /mnt/wslg/PulseServer ] && [ ! -f /etc/asound.conf ]; then printf 'pcm.!default { type pulse }\nctl.!default { type pulse }\n' > /etc/asound.conf && echo ALSA routed to WSLg PulseAudio; fi"
echo.
echo Done. Now run 9_LINUX_version_in_WSL to re-test.
echo Press any key to close this window.
pause >nul
