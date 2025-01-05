"""Script to start visualization test environment."""

import subprocess
import sys
import os
from pathlib import Path
import time
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def start_test():
    root_dir = Path(__file__).parent.parent
    logger.info(f"Starting from root directory: {root_dir}")

    logger.info("Starting WebSocket server...")
    server_cmd = [
        sys.executable,
        '-m', 'topo_explorer.visualization.websocket_server'
    ]
    server_process = subprocess.Popen(
        server_cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    logger.info("WebSocket server process started")

    time.sleep(2)

    logger.info("Starting frontend development server...")
    frontend_dir = root_dir / 'frontend'
    frontend_process = subprocess.Popen(
        ['npm', 'run', 'dev'],
        cwd=str(frontend_dir),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        universal_newlines=True,
        bufsize=1
    )
    logger.info("Frontend process started")

    def log_output(process, name):
        while True:
            line = process.stdout.readline()
            if not line and process.poll() is not None:
                break
            if line:
                logger.info(f"{name}: {line.strip()}")

    def log_errors(process, name):
        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break
            if line:
                logger.error(f"{name} ERROR: {line.strip()}")

    import threading
    for p, name in [(server_process, "Server"), (frontend_process, "Frontend")]:
        out_thread = threading.Thread(target=log_output, args=(p, name))
        err_thread = threading.Thread(target=log_errors, args=(p, name))
        out_thread.daemon = True
        err_thread.daemon = True
        out_thread.start()
        err_thread.start()

    try:
        while True:
            if server_process.poll() is not None:
                logger.error("WebSocket server stopped unexpectedly")
                frontend_process.terminate()
                break
            if frontend_process.poll() is not None:
                logger.error("Frontend server stopped unexpectedly")
                server_process.terminate()
                break
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down servers...")
        server_process.terminate()
        frontend_process.terminate()
        server_process.wait()
        frontend_process.wait()
        logger.info("Servers shut down")

if __name__ == '__main__':
    try:
        start_test()
    except Exception as e:
        logger.error(f"Error running test: {e}", exc_info=True)