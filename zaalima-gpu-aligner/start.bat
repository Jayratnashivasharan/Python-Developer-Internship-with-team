@echo off
REM One-command launcher (Windows): creates a virtual env, installs deps, starts the web app.
cd /d "%~dp0"
if not exist .venv (
  echo ^>^> creating virtual environment
  python -m venv .venv
)
call .venv\Scripts\activate.bat
echo ^>^> installing dependencies
python -m pip install -q --upgrade pip
python -m pip install -q -r requirements.txt
echo ^>^> starting web app on http://localhost:8000
python run_web.py
pause
