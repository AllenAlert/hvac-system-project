#!/usr/bin/env python3
"""
Run the HVAC dashboard. Installs dashboard deps if missing, then starts the server.
Usage: python run_dashboard.py
"""
import subprocess
import sys

def main():
    try:
        import fastapi
        import uvicorn
        import jinja2
    except ImportError:
        print("Installing dashboard dependencies (fastapi, uvicorn, jinja2, python-multipart)...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "fastapi", "uvicorn", "jinja2", "python-multipart", "-q"
        ])
        print("Done. Starting server...")

    import uvicorn
    uvicorn.run(
        "dashboard.main:app",
        host="127.0.0.1",
        port=8000,
        reload=False,
    )

if __name__ == "__main__":
    main()
