@echo on

cd /d %~dp0\

IF NOT EXIST ".venv" (
    python -m venv .venv

    call .venv\Scripts\activate.bat

    pip install -r .\requirements.txt
)

call .venv/Scripts/Activate.bat

start "" http://127.0.0.1:8000

fastapi run Frontend/main.py
