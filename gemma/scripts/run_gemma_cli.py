
import os
import sys
import subprocess

def main():
    # Get the directory of the script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Construct the path to the virtual environment's Python interpreter
    venv_python = os.path.join(script_dir, '..', '.venv', 'Scripts', 'python.exe')
    
    # Construct the path to the gemma-cli script
    gemma_cli_script = os.path.join(script_dir, '..', 'src', 'gemma_cli', 'cli.py')
    
    # Command to execute
    command = [venv_python, gemma_cli_script] + sys.argv[1:]
    
    # Execute the command
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)
    except FileNotFoundError:
        print(f"Error: Could not find {venv_python} or {gemma_cli_script}")
        sys.exit(1)

if __name__ == "__main__":
    main()
