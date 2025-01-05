"""Script to start both Flask and frontend development servers."""

import subprocess
import sys
import os
from pathlib import Path
import time

def start_servers():
    root_dir = Path(__file__).parent.parent

    print("Starting Flask server...")
    flask_process = subprocess.Popen([
        sys.executable,
        "-c",
        "from topo_explorer.server import run_server; run_server()"
    ])

    print("Starting frontend development server...")
    frontend_dir = root_dir / 'frontend'
    frontend_process = subprocess.Popen([
        'npm',
        'run',
        'dev'
    ], cwd=str(frontend_dir))

    try:
        while True:
            if flask_process.poll() is not None:
                print("Flask server stopped unexpectedly")
                frontend_process.terminate()
                break
            if frontend_process.poll() is not None:
                print("Frontend server stopped unexpectedly")
                flask_process.terminate()
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nShutting down servers...")
        flask_process.terminate()
        frontend_process.terminate()
        flask_process.wait()
        frontend_process.wait()

if __name__ == '__main__':
    start_servers()
