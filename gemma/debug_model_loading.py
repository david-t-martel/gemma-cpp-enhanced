#!/usr/bin/env python3
"""
Gemma Model Loading Diagnostic Tool
Systematically diagnose model loading issues and Windows crashes.
"""

import os
import sys
import subprocess
import json
import hashlib
import logging
from pathlib import Path
from datetime import datetime
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_loading_debug.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelLoadingDiagnostic:
    def __init__(self):
        self.models_dir = Path("C:/codedev/llm/.models")
        self.build_dir = Path("C:/codedev/llm/gemma/gemma.cpp/build")
        self.build_wsl_dir = Path("C:/codedev/llm/gemma/gemma.cpp/build_wsl")
        self.build_vs_dir = Path("C:/codedev/llm/gemma/gemma.cpp/build_vs")

        # Available models
        self.models = {
            "2b": {
                "weights": self.models_dir / "gemma-gemmacpp-2b-it-v3/2b-it.sbs",
                "tokenizer": self.models_dir / "gemma-gemmacpp-2b-it-v3/tokenizer.spm"
            },
            "4b": {
                "weights": self.models_dir / "gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/4b-it-sfp.sbs",
                "tokenizer": self.models_dir / "gemma-3-gemmaCpp-3.0-4b-it-sfp-v1/tokenizer.spm"
            }
        }

        self.results = {
            "timestamp": datetime.now().isoformat(),
            "tests": {},
            "environment": {},
            "recommendations": []
        }

    def check_environment(self):
        """Check basic environment setup"""
        logger.info("=== Environment Check ===")

        # Check directories
        env_info = {}
        env_info["models_dir_exists"] = self.models_dir.exists()
        env_info["build_dir_exists"] = self.build_dir.exists()
        env_info["build_wsl_dir_exists"] = self.build_wsl_dir.exists()
        env_info["build_vs_dir_exists"] = self.build_vs_dir.exists()

        # Check executables
        executables = {
            "unix_gemma": self.build_dir / "gemma",
            "wsl_gemma": self.build_wsl_dir / "gemma",
            "vs_gemma_debug": self.build_vs_dir / "x64/Debug/gemma.exe",
            "vs_gemma_release": self.build_vs_dir / "x64/Release/gemma.exe",
        }

        for name, path in executables.items():
            env_info[f"{name}_exists"] = path.exists()
            if path.exists():
                env_info[f"{name}_size"] = path.stat().st_size
                env_info[f"{name}_modified"] = datetime.fromtimestamp(path.stat().st_mtime).isoformat()

        # Check WSL availability
        try:
            result = subprocess.run(['wsl', '--version'], capture_output=True, text=True, timeout=10)
            env_info["wsl_available"] = result.returncode == 0
            env_info["wsl_version"] = result.stdout.strip() if result.returncode == 0 else None
        except Exception as e:
            env_info["wsl_available"] = False
            env_info["wsl_error"] = str(e)

        # Check Visual C++ Redistributables
        try:
            # Check for common VC++ redist registry entries
            import winreg
            redist_info = {}
            try:
                with winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, r"SOFTWARE\Microsoft\VisualStudio\14.0\VC\Runtimes\x64") as key:
                    redist_info["vcredist_2015_x64"] = True
            except FileNotFoundError:
                redist_info["vcredist_2015_x64"] = False

            env_info["vcredist"] = redist_info
        except Exception as e:
            env_info["vcredist_error"] = str(e)

        self.results["environment"] = env_info
        logger.info(f"Environment check completed: {json.dumps(env_info, indent=2)}")

        return env_info

    def check_model_integrity(self):
        """Check model file integrity"""
        logger.info("=== Model Integrity Check ===")

        integrity_results = {}

        for model_name, model_info in self.models.items():
            model_result = {}

            # Check if files exist
            weights_path = model_info["weights"]
            tokenizer_path = model_info["tokenizer"]

            model_result["weights_exists"] = weights_path.exists()
            model_result["tokenizer_exists"] = tokenizer_path.exists()

            if weights_path.exists():
                # File size and basic info
                stat = weights_path.stat()
                model_result["weights_size"] = stat.st_size
                model_result["weights_size_mb"] = round(stat.st_size / (1024*1024), 2)
                model_result["weights_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

                # Calculate MD5 hash for first 1MB to check corruption
                try:
                    with open(weights_path, 'rb') as f:
                        first_chunk = f.read(1024 * 1024)  # First 1MB
                        md5_hash = hashlib.md5(first_chunk).hexdigest()
                        model_result["weights_first_mb_md5"] = md5_hash
                except Exception as e:
                    model_result["weights_hash_error"] = str(e)

                # Check if file is readable
                try:
                    with open(weights_path, 'rb') as f:
                        # Try to read first few bytes to ensure file isn't corrupted
                        header = f.read(64)
                        model_result["weights_readable"] = True
                        model_result["weights_header_hex"] = header.hex()[:32]  # First 16 bytes as hex
                except Exception as e:
                    model_result["weights_readable"] = False
                    model_result["weights_read_error"] = str(e)

            if tokenizer_path.exists():
                stat = tokenizer_path.stat()
                model_result["tokenizer_size"] = stat.st_size
                model_result["tokenizer_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

                try:
                    with open(tokenizer_path, 'rb') as f:
                        header = f.read(64)
                        model_result["tokenizer_readable"] = True
                        model_result["tokenizer_header_hex"] = header.hex()[:32]
                except Exception as e:
                    model_result["tokenizer_readable"] = False
                    model_result["tokenizer_read_error"] = str(e)

            integrity_results[model_name] = model_result
            logger.info(f"Model {model_name} integrity: {json.dumps(model_result, indent=2)}")

        self.results["model_integrity"] = integrity_results
        return integrity_results

    def test_executable_basic(self, executable_path, use_wsl=False):
        """Test if executable runs with basic arguments"""
        logger.info(f"Testing executable: {executable_path} (WSL: {use_wsl})")

        if not executable_path.exists():
            return {"error": "Executable not found", "exists": False}

        result = {
            "exists": True,
            "size": executable_path.stat().st_size,
            "tests": {}
        }

        # Test 1: Help command
        try:
            if use_wsl:
                cmd = ['wsl', str(executable_path), '--help']
            else:
                cmd = [str(executable_path), '--help']

            proc = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            result["tests"]["help"] = {
                "returncode": proc.returncode,
                "stdout_length": len(proc.stdout),
                "stderr_length": len(proc.stderr),
                "success": proc.returncode == 0,
                "output_preview": proc.stdout[:200] if proc.stdout else proc.stderr[:200]
            }
        except subprocess.TimeoutExpired:
            result["tests"]["help"] = {"error": "timeout"}
        except Exception as e:
            result["tests"]["help"] = {"error": str(e)}

        # Test 2: Version or minimal run (if help fails)
        if not result["tests"]["help"].get("success", False):
            try:
                if use_wsl:
                    cmd = ['wsl', str(executable_path)]
                else:
                    cmd = [str(executable_path)]

                # Run with very short timeout to see if it crashes immediately
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
                result["tests"]["minimal_run"] = {
                    "returncode": proc.returncode,
                    "stdout_length": len(proc.stdout),
                    "stderr_length": len(proc.stderr),
                    "output_preview": (proc.stdout + proc.stderr)[:200]
                }
            except subprocess.TimeoutExpired:
                result["tests"]["minimal_run"] = {"result": "timeout_expected"}
            except Exception as e:
                result["tests"]["minimal_run"] = {"error": str(e)}

        return result

    def test_model_loading(self, model_name="2b", use_wsl=True):
        """Test actual model loading with different executables"""
        logger.info(f"=== Testing Model Loading: {model_name} ===")

        if model_name not in self.models:
            return {"error": f"Unknown model: {model_name}"}

        model_info = self.models[model_name]
        weights_path = model_info["weights"]
        tokenizer_path = model_info["tokenizer"]

        if not weights_path.exists() or not tokenizer_path.exists():
            return {"error": "Model files not found"}

        loading_results = {}

        # Test different executables
        executables_to_test = [
            ("unix_build", self.build_dir / "gemma", False),
            ("wsl_build", self.build_wsl_dir / "gemma", True),
        ]

        # Add VS builds if they exist
        vs_debug = self.build_vs_dir / "x64/Debug/gemma.exe"
        vs_release = self.build_vs_dir / "x64/Release/gemma.exe"
        if vs_debug.exists():
            executables_to_test.append(("vs_debug", vs_debug, False))
        if vs_release.exists():
            executables_to_test.append(("vs_release", vs_release, False))

        for build_name, exe_path, use_wsl_flag in executables_to_test:
            logger.info(f"Testing {build_name}: {exe_path}")

            if not exe_path.exists():
                loading_results[build_name] = {"error": "Executable not found"}
                continue

            try:
                # Prepare command
                if use_wsl_flag:
                    cmd = [
                        'wsl', str(exe_path),
                        '--tokenizer', str(tokenizer_path),
                        '--weights', str(weights_path),
                        '--prompt', 'Hi'
                    ]
                else:
                    cmd = [
                        str(exe_path),
                        '--tokenizer', str(tokenizer_path),
                        '--weights', str(weights_path),
                        '--prompt', 'Hi'
                    ]

                start_time = time.time()

                # Run with timeout
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=60)

                end_time = time.time()
                duration = end_time - start_time

                loading_results[build_name] = {
                    "returncode": proc.returncode,
                    "duration_seconds": round(duration, 2),
                    "stdout_length": len(proc.stdout),
                    "stderr_length": len(proc.stderr),
                    "success": proc.returncode == 0,
                    "stdout_preview": proc.stdout[:500] if proc.stdout else "",
                    "stderr_preview": proc.stderr[:500] if proc.stderr else "",
                    "command": " ".join(cmd)
                }

                # Check for specific error patterns
                output_text = proc.stdout + proc.stderr
                if "access violation" in output_text.lower():
                    loading_results[build_name]["error_type"] = "access_violation"
                elif "heap corruption" in output_text.lower():
                    loading_results[build_name]["error_type"] = "heap_corruption"
                elif "segmentation fault" in output_text.lower():
                    loading_results[build_name]["error_type"] = "segmentation_fault"

                logger.info(f"{build_name} result: returncode={proc.returncode}, duration={duration:.2f}s")

            except subprocess.TimeoutExpired:
                loading_results[build_name] = {
                    "error": "timeout",
                    "duration_seconds": 60.0
                }
                logger.warning(f"{build_name} timed out")

            except Exception as e:
                loading_results[build_name] = {
                    "error": str(e),
                    "error_type": type(e).__name__
                }
                logger.error(f"{build_name} failed: {e}")

        self.results["tests"][f"model_loading_{model_name}"] = loading_results
        return loading_results

    def check_windows_event_log(self):
        """Check Windows Event Viewer for recent application crashes"""
        logger.info("=== Checking Windows Event Log ===")

        try:
            # Use PowerShell to query Windows Event Log for application errors
            ps_command = """
            Get-WinEvent -FilterHashtable @{LogName='Application'; Level=2; StartTime=(Get-Date).AddHours(-24)} |
            Where-Object {$_.Id -eq 1000 -or $_.Id -eq 1001} |
            Select-Object -First 10 TimeCreated, Id, LevelDisplayName, ProviderName, Message |
            ConvertTo-Json
            """

            result = subprocess.run(
                ['powershell', '-Command', ps_command],
                capture_output=True, text=True, timeout=30
            )

            if result.returncode == 0 and result.stdout.strip():
                try:
                    events = json.loads(result.stdout)
                    if not isinstance(events, list):
                        events = [events]  # Single event case

                    # Filter for gemma-related crashes
                    gemma_events = []
                    for event in events:
                        if 'gemma' in event.get('Message', '').lower():
                            gemma_events.append(event)

                    event_log_result = {
                        "total_app_errors": len(events),
                        "gemma_related_errors": len(gemma_events),
                        "recent_events": events[:5],  # First 5 events
                        "gemma_events": gemma_events
                    }
                except json.JSONDecodeError:
                    event_log_result = {
                        "error": "Could not parse event log JSON",
                        "raw_output": result.stdout[:500]
                    }
            else:
                event_log_result = {
                    "error": "No recent application errors found or PowerShell failed",
                    "returncode": result.returncode,
                    "stderr": result.stderr[:500]
                }

        except Exception as e:
            event_log_result = {
                "error": f"Failed to check event log: {str(e)}"
            }

        self.results["windows_event_log"] = event_log_result
        logger.info(f"Event log check: {json.dumps(event_log_result, indent=2)}")
        return event_log_result

    def generate_recommendations(self):
        """Generate recommendations based on test results"""
        logger.info("=== Generating Recommendations ===")

        recommendations = []

        # Check environment issues
        env = self.results.get("environment", {})
        if not env.get("wsl_available", False):
            recommendations.append({
                "priority": "high",
                "issue": "WSL not available",
                "recommendation": "Install WSL2 and ensure it's properly configured",
                "commands": ["wsl --install", "wsl --set-default-version 2"]
            })

        # Check for missing VC++ redistributables
        if "vcredist_error" in env:
            recommendations.append({
                "priority": "medium",
                "issue": "Could not check VC++ redistributables",
                "recommendation": "Install latest Visual C++ Redistributable for x64",
                "url": "https://aka.ms/vs/17/release/vc_redist.x64.exe"
            })

        # Check model integrity issues
        model_integrity = self.results.get("model_integrity", {})
        for model_name, model_info in model_integrity.items():
            if not model_info.get("weights_exists", False):
                recommendations.append({
                    "priority": "high",
                    "issue": f"Model {model_name} weights not found",
                    "recommendation": f"Download and extract {model_name} model to correct location",
                    "path": self.models[model_name]["weights"]
                })

            if not model_info.get("weights_readable", True):
                recommendations.append({
                    "priority": "high",
                    "issue": f"Model {model_name} weights file corrupted or unreadable",
                    "recommendation": "Re-download the model file, check disk space and permissions",
                    "error": model_info.get("weights_read_error", "Unknown error")
                })

        # Check for crash patterns in model loading tests
        for test_name, test_results in self.results.get("tests", {}).items():
            if "model_loading" in test_name:
                for build_name, build_result in test_results.items():
                    if build_result.get("returncode") != 0:
                        error_type = build_result.get("error_type", "unknown")
                        if error_type == "access_violation":
                            recommendations.append({
                                "priority": "high",
                                "issue": f"Access violation in {build_name}",
                                "recommendation": "Rebuild with debug symbols, check for memory alignment issues",
                                "build": build_name
                            })
                        elif error_type == "heap_corruption":
                            recommendations.append({
                                "priority": "high",
                                "issue": f"Heap corruption in {build_name}",
                                "recommendation": "Check for buffer overflows, use AddressSanitizer build",
                                "build": build_name
                            })

        self.results["recommendations"] = recommendations
        return recommendations

    def run_full_diagnostic(self):
        """Run complete diagnostic suite"""
        logger.info("Starting full Gemma model loading diagnostic...")

        # Run all checks
        self.check_environment()
        self.check_model_integrity()
        self.check_windows_event_log()

        # Test basic executable functionality
        exe_tests = {}
        executables = [
            ("unix_build", self.build_dir / "gemma", False),
            ("wsl_build", self.build_wsl_dir / "gemma", True),
        ]

        for name, path, use_wsl in executables:
            exe_tests[name] = self.test_executable_basic(path, use_wsl)

        self.results["executable_tests"] = exe_tests

        # Test model loading with available models
        for model_name in self.models.keys():
            if self.models[model_name]["weights"].exists():
                self.test_model_loading(model_name)

        # Generate recommendations
        self.generate_recommendations()

        # Save results
        results_file = f"model_loading_diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(results_file, 'w') as f:
            json.dump(self.results, f, indent=2)

        logger.info(f"Diagnostic complete. Results saved to: {results_file}")
        return self.results

    def print_summary(self):
        """Print a human-readable summary"""
        print("\n" + "="*60)
        print("GEMMA MODEL LOADING DIAGNOSTIC SUMMARY")
        print("="*60)

        # Environment summary
        env = self.results.get("environment", {})
        print(f"\n📁 Environment:")
        print(f"   Models directory: {'✓' if env.get('models_dir_exists') else '✗'}")
        print(f"   WSL available: {'✓' if env.get('wsl_available') else '✗'}")
        print(f"   Unix build: {'✓' if env.get('unix_gemma_exists') else '✗'}")
        print(f"   WSL build: {'✓' if env.get('wsl_gemma_exists') else '✗'}")

        # Model integrity
        model_integrity = self.results.get("model_integrity", {})
        print(f"\n📊 Model Integrity:")
        for model_name, model_info in model_integrity.items():
            status = "✓" if model_info.get("weights_readable") and model_info.get("tokenizer_exists") else "✗"
            size_mb = model_info.get("weights_size_mb", 0)
            print(f"   {model_name.upper()} model: {status} ({size_mb:.1f} MB)")

        # Test results summary
        print(f"\n🧪 Test Results:")
        for test_name, test_results in self.results.get("tests", {}).items():
            if "model_loading" in test_name:
                model = test_name.split("_")[-1].upper()
                print(f"   {model} Model Loading:")
                for build_name, result in test_results.items():
                    if "error" in result:
                        print(f"     {build_name}: ✗ ({result.get('error', 'unknown error')})")
                    elif result.get("success"):
                        duration = result.get("duration_seconds", 0)
                        print(f"     {build_name}: ✓ ({duration:.1f}s)")
                    else:
                        code = result.get("returncode", "unknown")
                        print(f"     {build_name}: ✗ (exit code: {code})")

        # Recommendations
        recommendations = self.results.get("recommendations", [])
        if recommendations:
            print(f"\n💡 Recommendations ({len(recommendations)} issues found):")
            for i, rec in enumerate(recommendations[:5], 1):  # Show top 5
                priority = rec.get("priority", "medium").upper()
                issue = rec.get("issue", "Unknown issue")
                recommendation = rec.get("recommendation", "No recommendation")
                print(f"   {i}. [{priority}] {issue}")
                print(f"      → {recommendation}")
        else:
            print(f"\n✓ No critical issues found!")

        print("\n" + "="*60)

def main():
    """Main diagnostic function"""
    print("Gemma Model Loading Diagnostic Tool")
    print("===================================")

    diagnostic = ModelLoadingDiagnostic()

    try:
        results = diagnostic.run_full_diagnostic()
        diagnostic.print_summary()

        return 0 if not results.get("recommendations") else 1

    except KeyboardInterrupt:
        logger.info("Diagnostic cancelled by user")
        return 130
    except Exception as e:
        logger.error(f"Diagnostic failed: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())