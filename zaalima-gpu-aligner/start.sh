#!/usr/bin/env bash
# One-command launcher (Linux / macOS): creates a virtual env, installs deps, starts the web app.
set -e
cd "$(dirname "$0")"
PY=${PYTHON:-python3}
if [ ! -d .venv ]; then
  echo ">> creating virtual environment (.venv)"
  $PY -m venv .venv
fi
. .venv/bin/activate
echo ">> installing dependencies"
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
# Optional speed-up; the app works fine without it, so never fail here.
python setup.py build_ext --inplace >/dev/null 2>&1 && echo ">> Cython core built" || echo ">> (Cython build skipped - using Numba engine)"
echo ">> starting web app on http://localhost:${PORT:-8000}"
exec python run_web.py --host "${HOST:-127.0.0.1}" --port "${PORT:-8000}"
