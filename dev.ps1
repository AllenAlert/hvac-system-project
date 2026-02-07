#!/usr/bin/env pwsh
# Quick dev server with hot reload
Set-Location $PSScriptRoot
python -m uvicorn dashboard.main:app --reload --host 127.0.0.1 --port 8000
