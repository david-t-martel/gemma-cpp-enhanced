
import time
from pathlib import Path
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler

from ruff_formatter import apply_ruff_fixes

class RuffEventHandler(FileSystemEventHandler):
    def on_modified(self, event):
        if event.is_directory:
            return
        filepath = Path(event.src_path)
        if filepath.suffix == ".py":
            print(f"Detected change in {filepath}. Running ruff --fix...")
            result = apply_ruff_fixes(filepath)
            if result["success"]:
                print(f"Ruff --fix applied to {filepath}")
                if result["stdout"]:
                    print("Ruff stdout:")
                    print(result["stdout"])
            else:
                print(f"Error applying ruff --fix to {filepath}: {result["message"]}")
                if "stdout" in result:
                    print("Ruff stdout:")
                    print(result["stdout"])
                if "stderr" in result:
                    print("Ruff stderr:")
                    print(result["stderr"])

def start_file_watcher(path: Path):
    event_handler = RuffEventHandler()
    observer = Observer()
    observer.schedule(event_handler, path, recursive=True)
    observer.start()
    print(f"Watching for Python file changes in {path}. Press Ctrl+C to stop.")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

if __name__ == "__main__":
    # Watch the current working directory
    start_file_watcher(Path("."))
