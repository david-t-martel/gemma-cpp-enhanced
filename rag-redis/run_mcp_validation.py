#!/usr/bin/env python3
"""
MCP Validation Test Runner for RAG-Redis System

This script orchestrates the complete validation process:
1. Checks prerequisites (Redis, server binary)
2. Builds server binary if needed
3. Runs comprehensive Python validation
4. Runs Claude CLI integration tests
5. Generates consolidated report
"""

import subprocess
import sys
import os
import json
import time
from pathlib import Path
from datetime import datetime

# Colors for output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    PURPLE = '\033[95m'
    CYAN = '\033[96m'
    WHITE = '\033[97m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    END = '\033[0m'

class MCPValidationRunner:
    """Orchestrates the complete MCP validation process"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.validation_script = self.project_root / "validate_mcp.py"
        self.integration_script = self.project_root / "test_claude_integration.sh"
        self.server_binary = self.project_root / "rag-redis-system" / "mcp-server" / "target" / "release" / "mcp-server.exe"
        self.cargo_toml = self.project_root / "rag-redis-system" / "mcp-server" / "Cargo.toml"
        self.results = {}
        
    def _print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{title.center(80)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")
    
    def _print_step(self, step: str):
        """Print step message"""
        print(f"{Colors.CYAN}[STEP] {step}{Colors.END}")
    
    def _print_success(self, message: str):
        """Print success message"""
        print(f"{Colors.GREEN}✓ {message}{Colors.END}")
    
    def _print_error(self, message: str):
        """Print error message"""
        print(f"{Colors.RED}✗ {message}{Colors.END}")
    
    def _print_warning(self, message: str):
        """Print warning message"""
        print(f"{Colors.YELLOW}⚠ {message}{Colors.END}")
    
    def _print_info(self, message: str):
        """Print info message"""
        print(f"{Colors.BLUE}ℹ {message}{Colors.END}")
    
    def check_prerequisites(self) -> bool:
        """Check all prerequisites for validation"""
        self._print_header("Checking Prerequisites")
        
        prerequisites_ok = True
        
        # Check if we're in the right directory
        if not self.project_root.exists():
            self._print_error(f"Project root not found: {self.project_root}")
            return False
        self._print_success(f"Project root found: {self.project_root}")
        
        # Check validation scripts
        if not self.validation_script.exists():
            self._print_error(f"Validation script not found: {self.validation_script}")
            prerequisites_ok = False
        else:
            self._print_success("Python validation script found")
        
        if not self.integration_script.exists():
            self._print_error(f"Integration script not found: {self.integration_script}")
            prerequisites_ok = False
        else:
            self._print_success("Shell integration script found")
        
        # Check Redis connection
        try:
            result = subprocess.run(
                ["redis-cli", "-p", "6380", "ping"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0 and result.stdout.strip() == "PONG":
                self._print_success("Redis server is running on port 6380")
            else:
                self._print_error("Redis server not responding on port 6380")
                self._print_info("Start Redis with: redis-server --port 6380")
                prerequisites_ok = False
        except Exception as e:
            self._print_error(f"Failed to check Redis: {e}")
            prerequisites_ok = False
        
        # Check Rust/Cargo installation
        try:
            result = subprocess.run(
                ["cargo", "--version"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                self._print_success(f"Cargo found: {result.stdout.strip()}")
            else:
                self._print_error("Cargo not found")
                prerequisites_ok = False
        except Exception as e:
            self._print_error(f"Failed to check Cargo: {e}")
            prerequisites_ok = False
        
        # Check Python
        try:
            result = subprocess.run([sys.executable, "--version"], capture_output=True, text=True)
            self._print_success(f"Python found: {result.stdout.strip()}")
        except Exception as e:
            self._print_error(f"Python check failed: {e}")
            prerequisites_ok = False
        
        return prerequisites_ok
    
    def build_server_if_needed(self) -> bool:
        """Build MCP server binary if it doesn't exist or is outdated"""
        self._print_header("Building MCP Server")
        
        # Check if binary exists
        if self.server_binary.exists():
            # Check if binary is newer than source
            binary_mtime = self.server_binary.stat().st_mtime
            cargo_mtime = self.cargo_toml.stat().st_mtime if self.cargo_toml.exists() else 0
            
            if binary_mtime > cargo_mtime:
                self._print_success("MCP server binary is up to date")
                return True
            else:
                self._print_warning("MCP server binary is outdated, rebuilding...")
        else:
            self._print_step("MCP server binary not found, building...")
        
        # Build the server
        try:
            server_dir = self.project_root / "rag-redis-system" / "mcp-server"
            
            self._print_step("Building MCP server (this may take a few minutes)...")
            
            # Use cargo build --release
            result = subprocess.run(
                ["cargo", "build", "--release"],
                cwd=server_dir,
                capture_output=True,
                text=True,
                timeout=300  # 5 minutes timeout
            )
            
            if result.returncode == 0:
                self._print_success("MCP server built successfully")
                return True
            else:
                self._print_error("Failed to build MCP server")
                print(f"STDOUT: {result.stdout}")
                print(f"STDERR: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            self._print_error("Build timed out after 5 minutes")
            return False
        except Exception as e:
            self._print_error(f"Build failed with error: {e}")
            return False
    
    def run_python_validation(self) -> bool:
        """Run the comprehensive Python validation script"""
        self._print_header("Running Python Validation")
        
        try:
            self._print_step("Executing comprehensive MCP validation...")
            
            start_time = time.time()
            result = subprocess.run(
                [sys.executable, str(self.validation_script), str(self.project_root)],
                capture_output=True,
                text=True,
                timeout=600  # 10 minutes timeout
            )
            duration = time.time() - start_time
            
            self.results['python_validation'] = {
                'returncode': result.returncode,
                'duration_s': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            print(result.stdout)
            if result.stderr:
                print(f"{Colors.YELLOW}STDERR:{Colors.END}\n{result.stderr}")
            
            if result.returncode == 0:
                self._print_success(f"Python validation completed successfully in {duration:.1f}s")
                return True
            elif result.returncode == 2:
                self._print_warning(f"Python validation completed with non-critical failures in {duration:.1f}s")
                return True
            else:
                self._print_error(f"Python validation failed with return code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            self._print_error("Python validation timed out after 10 minutes")
            return False
        except Exception as e:
            self._print_error(f"Python validation failed with error: {e}")
            return False
    
    def run_claude_integration(self) -> bool:
        """Run Claude CLI integration tests"""
        self._print_header("Running Claude CLI Integration Tests")
        
        # Check if bash is available (for Windows with WSL or Git Bash)
        bash_cmd = None
        for cmd in ["bash", "wsl", "git-bash"]:
            try:
                subprocess.run([cmd, "--version"], capture_output=True, timeout=5)
                bash_cmd = cmd
                break
            except:
                continue
        
        if not bash_cmd:
            self._print_warning("Bash not available, skipping Claude CLI integration tests")
            self._print_info("Install WSL or Git Bash to run these tests")
            return True
        
        try:
            self._print_step("Executing Claude CLI integration tests...")
            
            start_time = time.time()
            
            # Convert Windows path to WSL path if using WSL
            script_path = str(self.integration_script)
            if bash_cmd == "wsl":
                script_path = script_path.replace("c:\\", "/mnt/c/").replace("\\", "/")
            
            result = subprocess.run(
                [bash_cmd, script_path],
                capture_output=True,
                text=True,
                timeout=900  # 15 minutes timeout
            )
            duration = time.time() - start_time
            
            self.results['claude_integration'] = {
                'returncode': result.returncode,
                'duration_s': duration,
                'stdout': result.stdout,
                'stderr': result.stderr
            }
            
            print(result.stdout)
            if result.stderr:
                print(f"{Colors.YELLOW}STDERR:{Colors.END}\n{result.stderr}")
            
            if result.returncode == 0:
                self._print_success(f"Claude integration tests completed successfully in {duration:.1f}s")
                return True
            else:
                self._print_error(f"Claude integration tests failed with return code {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            self._print_error("Claude integration tests timed out after 15 minutes")
            return False
        except Exception as e:
            self._print_error(f"Claude integration tests failed with error: {e}")
            return False
    
    def generate_consolidated_report(self):
        """Generate a consolidated test report"""
        self._print_header("Generating Consolidated Report")
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "project_root": str(self.project_root),
            "validation_results": self.results,
            "summary": {
                "python_validation_passed": self.results.get('python_validation', {}).get('returncode', -1) <= 2,
                "claude_integration_passed": self.results.get('claude_integration', {}).get('returncode', -1) == 0,
                "total_duration_s": sum(
                    result.get('duration_s', 0) for result in self.results.values()
                )
            }
        }
        
        # Determine overall status
        python_ok = report['summary']['python_validation_passed']
        claude_ok = report['summary']['claude_integration_passed']
        
        if python_ok and claude_ok:
            report['summary']['overall_status'] = 'SUCCESS'
        elif python_ok:
            report['summary']['overall_status'] = 'PARTIAL_SUCCESS'
        else:
            report['summary']['overall_status'] = 'FAILURE'
        
        # Save report
        report_file = self.project_root / "mcp_validation_consolidated_report.json"
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, default=str)
        
        self._print_success(f"Consolidated report saved to: {report_file}")
        
        # Print summary
        print(f"\n{Colors.BOLD}=== VALIDATION SUMMARY ==={Colors.END}")
        print(f"Total Duration: {report['summary']['total_duration_s']:.1f}s")
        print(f"Python Validation: {'✓ PASS' if python_ok else '✗ FAIL'}")
        print(f"Claude Integration: {'✓ PASS' if claude_ok else '✗ FAIL'}")
        print(f"Overall Status: {report['summary']['overall_status']}")
        
        return report['summary']['overall_status'] == 'SUCCESS'
    
    def run_complete_validation(self) -> bool:
        """Run the complete validation process"""
        self._print_header("RAG-Redis MCP Validation Test Runner")
        
        self._print_info(f"Starting validation for project: {self.project_root}")
        self._print_info(f"Timestamp: {datetime.now().isoformat()}")
        
        # Step 1: Check prerequisites
        if not self.check_prerequisites():
            self._print_error("Prerequisites check failed")
            return False
        
        # Step 2: Build server if needed
        if not self.build_server_if_needed():
            self._print_error("Server build failed")
            return False
        
        # Step 3: Run Python validation
        python_ok = self.run_python_validation()
        
        # Step 4: Run Claude integration (even if Python failed)
        claude_ok = self.run_claude_integration()
        
        # Step 5: Generate report
        overall_success = self.generate_consolidated_report()
        
        if overall_success:
            self._print_success("Complete validation passed!")
        elif python_ok or claude_ok:
            self._print_warning("Validation completed with some issues")
        else:
            self._print_error("Validation failed")
        
        return overall_success

def main():
    """Main function"""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <project_root>")
        print(f"Example: {sys.argv[0]} c:\\codedev\\llm\\rag-redis")
        sys.exit(1)
    
    project_root = sys.argv[1]
    runner = MCPValidationRunner(project_root)
    
    try:
        success = runner.run_complete_validation()
        sys.exit(0 if success else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Validation failed with unexpected error: {e}{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    main()