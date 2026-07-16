import os
import sys
import subprocess
import webbrowser
import time

def log(msg):
    print(f"\n>>> [BOOTSTRAP] {msg}\n")

def main():
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 1. Determine virtual environment paths
    venv_dir = os.path.join(workspace_dir, ".venv")
    if sys.platform.startswith("win"):
        python_bin = os.path.join(venv_dir, "Scripts", "python.exe")
        pip_bin = os.path.join(venv_dir, "Scripts", "pip.exe")
    else:
        python_bin = os.path.join(venv_dir, "bin", "python")
        pip_bin = os.path.join(venv_dir, "bin", "pip")

    # 2. Create virtual environment if missing
    if not os.path.exists(venv_dir):
        log("Creating Python Virtual Environment (venv) in .venv...")
        try:
            subprocess.run([sys.executable, "-m", "venv", venv_dir], check=True)
            log("Virtual environment created successfully.")
        except Exception as e:
            log(f"Failed to create virtual environment: {e}. Using default system python.")
            python_bin = sys.executable
            pip_bin = "pip"

    # 3. Upgrade pip and install requirements
    log("Installing dependencies from requirements.txt...")
    try:
        req_file = os.path.join(workspace_dir, "requirements.txt")
        # Upgrade pip first
        subprocess.run([python_bin, "-m", "pip", "install", "--upgrade", "pip"], check=True)
        # Install packages
        subprocess.run([pip_bin, "install", "-r", req_file], check=True)
        log("Dependencies installed successfully.")
    except Exception as e:
        log(f"Dependency installation failed: {e}. Check your internet connection.")
        sys.exit(1)

    # 4. Open browser in a separate process after a small delay
    url = "http://127.0.0.1:8000"
    log(f"Server starting. Opening web browser to {url}...")
    
    # Simple delay loop before opening browser to ensure uvicorn finishes launching
    def launch_browser():
        time.sleep(2.5)
        webbrowser.open(url)
    
    # Launch browser thread
    import threading
    threading.Thread(target=launch_browser, daemon=True).start()

    # 5. Start Uvicorn Server using the virtual environment's python environment
    # We run it via subprocess to ensure standard IO behaviors
    log("Starting EduPredict AI Web Server on http://127.0.0.1:8000")
    try:
        cmd = [python_bin, "-m", "uvicorn", "app.main:app", "--host", "127.0.0.1", "--port", "8000"]
        subprocess.run(cmd, cwd=workspace_dir, check=True)
    except KeyboardInterrupt:
        log("Server stopped by user.")
    except Exception as e:
        log(f"Failed to start uvicorn: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
