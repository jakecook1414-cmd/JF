param()

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $root

if (-not (Test-Path -Path '.venv')) {
    python -m venv .venv
}

$python = Join-Path $root '.venv\Scripts\python.exe'

try {
    & $python -c "import importlib; importlib.import_module('streamlit')"
} catch {
    & $python -m pip install -r requirements.txt
}

& $python -m streamlit run app.py
