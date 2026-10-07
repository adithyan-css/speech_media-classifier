#!/usr/bin/env bash
# ======================================================================
#  ai_rf2 - one-click launcher for Linux / WSL / (later) ODROID N2+
#     bash demo_linux.sh          -> menu
#     bash demo_linux.sh 5        -> run menu item 5 directly
# ======================================================================
set -u
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE"
export PYTHONPATH="$HERE/src${PYTHONPATH:+:$PYTHONPATH}"
export PYTHONDONTWRITEBYTECODE=1
export AI_RF2_DATA="${AI_RF2_DATA:-$HOME/ai_rf2_data}"
mkdir -p "$AI_RF2_DATA"
BACKEND="torch"

has_torch() { "$@" -c "import torch" >/dev/null 2>&1; }

# ---------- 1. find a Python that already has PyTorch ----------
PY=()
if [ -n "${AI_RF2_PYTHON:-}" ]; then PY=("$AI_RF2_PYTHON"); fi
if [ ${#PY[@]} -eq 0 ] && [ -x "$HOME/.venvs/ai_rf2/bin/python" ] && has_torch "$HOME/.venvs/ai_rf2/bin/python"; then
  PY=("$HOME/.venvs/ai_rf2/bin/python"); fi
if [ ${#PY[@]} -eq 0 ] && [ -x "$HERE/../AAC_RF2/.venv/bin/python" ] && has_torch "$HERE/../AAC_RF2/.venv/bin/python"; then
  PY=("$HERE/../AAC_RF2/.venv/bin/python"); fi
if [ ${#PY[@]} -eq 0 ] && command -v python3 >/dev/null && has_torch python3; then PY=(python3); fi
if [ ${#PY[@]} -eq 0 ] && command -v python >/dev/null && has_torch python; then PY=(python); fi
if [ ${#PY[@]} -eq 0 ] && command -v python3 >/dev/null; then PY=(python3); fi
if [ ${#PY[@]} -eq 0 ]; then
  echo "Python 3.10+ not found. Ubuntu / ODROID:  sudo apt install python3 python3-venv python3-pip"
  exit 1
fi

# ---------- 2. make sure every package is there (installs only what is missing) ----------
MARK="$AI_RF2_DATA/setup_ok.txt"
if [ ! -f "$MARK" ] || [ "$(cat "$MARK")" != "${PY[*]}" ]; then
  echo "Checking the Python environment - first run only ..."
  "${PY[@]}" "$HERE/scripts/check_env.py" --install
  rc=$?
  if [ $rc -eq 3 ]; then
    # 'externally managed' system Python (Ubuntu 23.04+): use a venv that still sees PyTorch
    echo "Creating ~/.venvs/ai_rf2 (keeps access to the PyTorch you already have) ..."
    if "${PY[@]}" -m venv --system-site-packages "$HOME/.venvs/ai_rf2" 2>/dev/null; then
      PY=("$HOME/.venvs/ai_rf2/bin/python")
    else
      echo "python3-venv is not installed - installing packages for this user instead."
      rm -rf "$HOME/.venvs/ai_rf2"
      export AI_RF2_PIP_FLAGS="--user --break-system-packages"
    fi
    "${PY[@]}" "$HERE/scripts/check_env.py" --install; rc=$?
  fi
  if [ $rc -ne 0 ]; then echo "Setup did not finish - see the messages above."; exit 1; fi
  echo "${PY[*]}" > "$MARK"
fi

open_folder() { command -v xdg-open >/dev/null && xdg-open "$1" >/dev/null 2>&1 || echo "pictures are in: $1"; }

run_choice() {
  local c="$1"; local f="${2:-}"
  case "$c" in
    1) "${PY[@]}" -m ai_rf2.live --programme 0 --backend "$BACKEND" ${AI_RF2_LIVE_ARGS:-} ;;
    2) "${PY[@]}" -m ai_rf2.live --programme 1 --backend "$BACKEND" ${AI_RF2_LIVE_ARGS:-} ;;
    3) [ -z "$f" ] && read -r -p "Path of an audio file: " f
       "${PY[@]}" -m ai_rf2.live --file "$f" --backend "$BACKEND" ${AI_RF2_LIVE_ARGS:-} ;;
    p|P) "${PY[@]}" -m ai_rf2.live --pick --backend "$BACKEND" ${AI_RF2_LIVE_ARGS:-} ;;
    4) "${PY[@]}" -m ai_rf2.live --mic --backend "$BACKEND" ${AI_RF2_LIVE_ARGS:-} ;;
    5) "${PY[@]}" -m ai_rf2.demo --backend "$BACKEND" --compare; local r=$?
       [ -z "$ONESHOT" ] && open_folder "$HERE/reports/demo"; return $r ;;
    6) [ -z "$f" ] && read -r -p "Path of an audio file: " f
       "${PY[@]}" -m ai_rf2.infer "$f" --backend "$BACKEND" ;;
    7) "${PY[@]}" -m pytest "$HERE/tests" -q -m "not slow" -p no:cacheprovider --junitxml="$HERE/reports/tests_linux.xml" ;;
    t|T) "${PY[@]}" "$HERE/scripts/selftest.py" ${AI_RF2_SELFTEST_ARGS:-} ;;
    8) "${PY[@]}" "$HERE/scripts/check_env.py" --install ;;
    9) echo "Big data goes to $AI_RF2_DATA"; "${PY[@]}" "$HERE/scripts/run_all.py" ;;
    b|B) if [ "$BACKEND" = torch ]; then BACKEND=onnx; else BACKEND=torch; fi ;;
    0) exit 0 ;;
    *) echo "unknown choice: $c" ;;
  esac
}

ONESHOT="${1:-}"
if [ -n "$ONESHOT" ]; then run_choice "$ONESHOT" "${2:-}"; exit $?; fi

while true; do
  clear 2>/dev/null
  echo "======================================================================"
  echo "  ai_rf2  -  AI Radio Announcement Classifier  (speech / music demo)"
  echo "======================================================================"
  echo "  Python : ${PY[*]}"
  echo "  Model  : $BACKEND    (torch = PyTorch model, onnx = 8-bit ODROID model)"
  echo
  echo "  1  LIVE demo   - radio programme 1  (sound + live window)"
  echo "  2  LIVE demo   - radio programme 2"
  echo "  3  LIVE demo   - your own audio file (mp3 / wav / flac)"
  echo "  P  LIVE demo   - CHOOSE a song / audio file on this computer (plays + classifies)"
  echo "  4  LIVE demo   - microphone or line-in"
  echo "  5  REPORT demo - both programmes: result table + timeline pictures"
  echo "  6  Classify one audio file (text only)"
  echo "  7  Run the automatic tests"
  echo "  8  Check the Python setup"
  echo "  9  Retrain everything from scratch (downloads 3.2 GB, needs ~12 GB disk)"
  echo "  T  Full self-test of this machine (about 5 minutes, silent)"
  echo "  B  Switch model  PyTorch <-> 8-bit ONNX"
  echo "  0  Exit"
  echo
  read -r -p "Type a number and press Enter: " choice
  run_choice "$choice"
  case "$choice" in b|B) continue ;; esac
  echo; read -r -p "Press Enter to return to the menu ... " _
done
