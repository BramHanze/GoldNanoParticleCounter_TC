@echo on

cd /d %~dp0\

call .venv/Scripts/Activate.bat

start "" http://127.0.0.1:8000

fastapi run Frontend/main.py
