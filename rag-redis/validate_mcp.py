#!/usr/bin/env python3
"""
Comprehensive MCP Validation Script for RAG-Redis System

This script validates the entire MCP server functionality including:
- MCP configuration validation
- Tool functionality tests
- Performance benchmarks
- Error handling tests
- Claude CLI integration
- Multi-agent compatibility
"""

import json
import subprocess
import asyncio
import time
import sys
import os
import tempfile
import statistics
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
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

@dataclass
class TestResult:
    """Test result data structure"""
    test_name: str
    passed: bool
    duration_ms: float
    error_message: Optional[str] = None
    performance_metrics: Optional[Dict[str, Any]] = None
    expected_output: Optional[Any] = None
    actual_output: Optional[Any] = None

@dataclass
class ValidationSummary:
    """Validation summary with statistics"""
    total_tests: int
    passed_tests: int
    failed_tests: int
    total_duration_s: float
    average_test_time_ms: float
    performance_benchmarks: Dict[str, float]
    critical_failures: List[str]
    warnings: List[str]

class MCPValidator:
    """Main MCP validation class"""
    
    def __init__(self, project_root: str):
        self.project_root = Path(project_root)
        self.mcp_config_path = self.project_root / "mcp.json"
        self.server_binary = self.project_root / "rag-redis-system" / "mcp-server" / "target" / "release" / "mcp-server.exe"
        self.results: List[TestResult] = []
        self.redis_url = "redis://127.0.0.1:6380"
        self.test_project_id = f"test-{int(time.time())}"
        
        # Load MCP configuration
        self.mcp_config = self._load_mcp_config()
        
    def _load_mcp_config(self) -> Dict[str, Any]:
        """Load and validate MCP configuration"""
        try:
            with open(self.mcp_config_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"{Colors.RED}ERROR: Failed to load MCP config: {e}{Colors.END}")
            sys.exit(1)
    
    def _print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{title.center(80)}{Colors.END}")
        print(f"{Colors.BOLD}{Colors.BLUE}{'='*80}{Colors.END}\n")
    
    def _print_test_start(self, test_name: str):
        """Print test start message"""
        print(f"{Colors.CYAN}[TEST] {test_name}...{Colors.END}", end=" ", flush=True)
    
    def _print_test_result(self, result: TestResult):
        """Print test result"""
        if result.passed:
            print(f"{Colors.GREEN}[OK] PASS{Colors.END} ({result.duration_ms:.1f}ms)")
            if result.performance_metrics:
                for metric, value in result.performance_metrics.items():
                    print(f"       {Colors.YELLOW}{metric}: {value}{Colors.END}")
        else:
            print(f"{Colors.RED}[FAIL] FAIL{Colors.END} ({result.duration_ms:.1f}ms)")
            if result.error_message:
                print(f"       {Colors.RED}Error: {result.error_message}{Colors.END}")
    
    async def _run_test(self, test_name: str, test_func) -> TestResult:
        """Run a single test and record results"""
        self._print_test_start(test_name)
        start_time = time.time()
        
        try:
            result = await test_func()
            if isinstance(result, TestResult):
                test_result = result
            elif isinstance(result, bool):
                test_result = TestResult(
                    test_name=test_name,
                    passed=result,
                    duration_ms=(time.time() - start_time) * 1000
                )
            else:
                test_result = TestResult(
                    test_name=test_name,
                    passed=True,
                    duration_ms=(time.time() - start_time) * 1000,
                    actual_output=result
                )
        except Exception as e:
            test_result = TestResult(
                test_name=test_name,
                passed=False,
                duration_ms=(time.time() - start_time) * 1000,
                error_message=str(e)
            )
        
        self._print_test_result(test_result)
        self.results.append(test_result)
        return test_result
    
    def _check_redis_connection(self) -> bool:
        """Check if Redis is running and accessible"""
        try:
            result = subprocess.run(
                ["redis-cli", "-p", "6380", "ping"],
                capture_output=True,
                text=True,
                timeout=5
            )
            return result.returncode == 0 and result.stdout.strip() == "PONG"
        except Exception:
            return False
    
    def _check_server_binary(self) -> bool:
        """Check if MCP server binary exists and is executable"""
        return self.server_binary.exists()
    
    async def _start_mcp_server(self) -> Optional[subprocess.Popen]:
        """Start MCP server for testing"""
        if not self._check_server_binary():
            raise Exception(f"MCP server binary not found: {self.server_binary}")
        
        env = os.environ.copy()
        env.update({
            "REDIS_URL": self.redis_url,
            "RUST_LOG": "debug",
            "RAG_DATA_DIR": str(self.project_root / "data" / "rag"),
            "EMBEDDING_CACHE_DIR": str(self.project_root / "cache" / "embeddings"),
            "LOG_DIR": str(self.project_root / "logs"),
        })
        
        process = subprocess.Popen(
            [str(self.server_binary)],
            env=env,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Give server time to start
        await asyncio.sleep(2)
        return process
    
    async def _send_mcp_request(self, process: subprocess.Popen, method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Send MCP request to server"""
        request = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": method,
            "params": params
        }
        
        request_json = json.dumps(request) + '\n'
        process.stdin.write(request_json)
        process.stdin.flush()
        
        # Read response
        response_line = process.stdout.readline()
        if not response_line:
            raise Exception("No response from server")
        
        return json.loads(response_line.strip())
    
    # ============================================================================
    # TEST METHODS
    # ============================================================================
    
    async def test_mcp_config_validation(self) -> TestResult:
        """Test MCP configuration file validation"""
        errors = []
        
        # Check required fields
        if "mcpServers" not in self.mcp_config:
            errors.append("Missing 'mcpServers' field")
        
        if "rag-redis" not in self.mcp_config.get("mcpServers", {}):
            errors.append("Missing 'rag-redis' server configuration")
        
        rag_config = self.mcp_config.get("mcpServers", {}).get("rag-redis", {})
        
        # Check command path
        command = rag_config.get("command")
        if not command or not Path(command).exists():
            errors.append(f"MCP server binary not found: {command}")
        
        # Check required environment variables
        env_vars = rag_config.get("env", {})
        required_env = ["REDIS_URL", "RAG_DATA_DIR", "EMBEDDING_CACHE_DIR", "LOG_DIR"]
        for var in required_env:
            if var not in env_vars:
                errors.append(f"Missing required environment variable: {var}")
        
        # Check capabilities
        capabilities = rag_config.get("capabilities", {})
        if "tools" not in capabilities:
            errors.append("Missing tools in capabilities")
        
        # Validate tool definitions
        tools = capabilities.get("tools", {})
        expected_tools = [
            "ingest_document", "search", "hybrid_search", "research",
            "memory_store", "memory_recall", "health_check",
            "project_context_save", "project_context_load",
            "agent_memory_store", "agent_memory_retrieve", "memory_digest"
        ]
        
        for tool in expected_tools:
            if tool not in tools:
                errors.append(f"Missing expected tool: {tool}")
        
        return TestResult(
            test_name="MCP Config Validation",
            passed=len(errors) == 0,
            duration_ms=0,
            error_message="; ".join(errors) if errors else None
        )
    
    async def test_redis_connectivity(self) -> TestResult:
        """Test Redis connection"""
        connected = self._check_redis_connection()
        
        return TestResult(
            test_name="Redis Connectivity",
            passed=connected,
            duration_ms=0,
            error_message="Redis server not accessible on port 6380" if not connected else None
        )
    
    async def test_server_binary_exists(self) -> TestResult:
        """Test MCP server binary existence"""
        exists = self._check_server_binary()
        
        return TestResult(
            test_name="Server Binary Exists",
            passed=exists,
            duration_ms=0,
            error_message=f"Server binary not found: {self.server_binary}" if not exists else None
        )
    
    async def test_server_startup(self) -> TestResult:
        """Test MCP server startup"""
        try:
            process = await self._start_mcp_server()
            if process.poll() is not None:
                # Process already terminated
                stderr = process.stderr.read()
                raise Exception(f"Server failed to start: {stderr}")
            
            # Test basic communication
            response = await self._send_mcp_request(process, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "validator", "version": "1.0.0"}
            })
            
            process.terminate()
            
            if "result" not in response:
                raise Exception(f"Invalid initialization response: {response}")
            
            return TestResult(
                test_name="Server Startup",
                passed=True,
                duration_ms=0
            )
            
        except Exception as e:
            return TestResult(
                test_name="Server Startup",
                passed=False,
                duration_ms=0,
                error_message=str(e)
            )
    
    async def test_health_check_tool(self) -> TestResult:
        """Test health_check tool"""
        try:
            process = await self._start_mcp_server()
            
            # Initialize
            await self._send_mcp_request(process, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "validator", "version": "1.0.0"}
            })
            
            # Call health check
            response = await self._send_mcp_request(process, "tools/call", {
                "name": "health_check",
                "arguments": {"include_metrics": True}
            })
            
            process.terminate()
            
            if "result" not in response:
                raise Exception(f"Health check failed: {response}")
            
            result_data = response["result"]
            if "content" not in result_data:
                raise Exception("Health check response missing content")
            
            return TestResult(
                test_name="Health Check Tool",
                passed=True,
                duration_ms=0,
                actual_output=result_data
            )
            
        except Exception as e:
            return TestResult(
                test_name="Health Check Tool",
                passed=False,
                duration_ms=0,
                error_message=str(e)
            )
    
    async def test_document_ingestion(self) -> TestResult:
        """Test document ingestion functionality"""
        try:
            process = await self._start_mcp_server()
            
            # Initialize
            await self._send_mcp_request(process, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "validator", "version": "1.0.0"}
            })
            
            # Ingest test document
            test_content = "This is a test document about machine learning and artificial intelligence."
            test_metadata = {
                "source": "validation_test",
                "title": "Test Document",
                "author": "MCP Validator",
                "timestamp": datetime.now().isoformat()
            }
            
            response = await self._send_mcp_request(process, "tools/call", {
                "name": "ingest_document",
                "arguments": {
                    "content": test_content,
                    "metadata": test_metadata
                }
            })
            
            process.terminate()
            
            if "result" not in response:
                raise Exception(f"Document ingestion failed: {response}")
            
            return TestResult(
                test_name="Document Ingestion",
                passed=True,
                duration_ms=0,
                actual_output=response["result"]
            )
            
        except Exception as e:
            return TestResult(
                test_name="Document Ingestion",
                passed=False,
                duration_ms=0,
                error_message=str(e)
            )
    
    async def test_semantic_search(self) -> TestResult:
        """Test semantic search functionality"""
        try:
            process = await self._start_mcp_server()
            
            # Initialize
            await self._send_mcp_request(process, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "validator", "version": "1.0.0"}
            })
            
            # First ingest a document
            await self._send_mcp_request(process, "tools/call", {
                "name": "ingest_document",
                "arguments": {
                    "content": "This document discusses machine learning algorithms and neural networks.",
                    "metadata": {"source": "test_search"}
                }
            })
            
            # Wait for indexing
            await asyncio.sleep(1)
            
            # Search for related content
            response = await self._send_mcp_request(process, "tools/call", {
                "name": "search",
                "arguments": {
                    "query": "machine learning algorithms",
                    "limit": 5,
                    "threshold": 0.5
                }
            })
            
            process.terminate()
            
            if "result" not in response:
                raise Exception(f"Search failed: {response}")
            
            return TestResult(
                test_name="Semantic Search",
                passed=True,
                duration_ms=0,
                actual_output=response["result"]
            )
            
        except Exception as e:
            return TestResult(
                test_name="Semantic Search",
                passed=False,
                duration_ms=0,
                error_message=str(e)
            )
    
    async def test_memory_operations(self) -> TestResult:
        """Test memory store and recall operations"""
        try:
            process = await self._start_mcp_server()
            
            # Initialize
            await self._send_mcp_request(process, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "validator", "version": "1.0.0"}
            })
            
            # Store memory
            test_memory = "The user prefers detailed explanations with examples."
            store_response = await self._send_mcp_request(process, "tools/call", {
                "name": "memory_store",
                "arguments": {
                    "content": test_memory,
                    "memory_type": "working",
                    "importance": 0.8,
                    "context_hints": ["user_preference", "explanation_style"]
                }
            })
            
            # Recall memory
            recall_response = await self._send_mcp_request(process, "tools/call", {
                "name": "memory_recall",
                "arguments": {
                    "query": "user preferences",
                    "memory_type": "working",
                    "limit": 5
                }
            })
            
            process.terminate()
            
            if "result" not in store_response or "result" not in recall_response:
                raise Exception("Memory operations failed")
            
            return TestResult(
                test_name="Memory Operations",
                passed=True,
                duration_ms=0,
                actual_output={
                    "store": store_response["result"],
                    "recall": recall_response["result"]
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="Memory Operations",
                passed=False,
                duration_ms=0,
                error_message=str(e)
            )
    
    async def test_agent_memory_integration(self) -> TestResult:
        """Test agent-specific memory operations"""
        try:
            process = await self._start_mcp_server()
            
            # Initialize
            await self._send_mcp_request(process, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "validator", "version": "1.0.0"}
            })
            
            # Store agent memory
            store_response = await self._send_mcp_request(process, "tools/call", {
                "name": "agent_memory_store",
                "arguments": {
                    "content": "Claude prefers structured responses with clear headings and bullet points.",
                    "agent_type": "claude",
                    "context_hints": ["response_format", "user_preference"],
                    "importance": 0.9
                }
            })
            
            # Retrieve agent memory
            retrieve_response = await self._send_mcp_request(process, "tools/call", {
                "name": "agent_memory_retrieve",
                "arguments": {
                    "query": "response format preferences",
                    "agent_type": "claude",
                    "limit": 5
                }
            })
            
            # Generate memory digest
            digest_response = await self._send_mcp_request(process, "tools/call", {
                "name": "memory_digest",
                "arguments": {
                    "agent_type": "claude",
                    "topic": "user_preferences",
                    "max_memories": 10
                }
            })
            
            process.terminate()
            
            return TestResult(
                test_name="Agent Memory Integration",
                passed=True,
                duration_ms=0,
                actual_output={
                    "store": store_response["result"],
                    "retrieve": retrieve_response["result"],
                    "digest": digest_response["result"]
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="Agent Memory Integration",
                passed=False,
                duration_ms=0,
                error_message=str(e)
            )
    
    async def test_project_context_operations(self) -> TestResult:
        """Test project context save and load"""
        try:
            process = await self._start_mcp_server()
            
            # Initialize
            await self._send_mcp_request(process, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "validator", "version": "1.0.0"}
            })
            
            # Save project context
            context_data = {
                "current_task": "MCP validation testing",
                "progress": "Running comprehensive tests",
                "next_steps": ["Performance benchmarks", "Error handling tests"],
                "key_findings": ["Server starts correctly", "Basic tools functional"]
            }
            
            save_response = await self._send_mcp_request(process, "tools/call", {
                "name": "project_context_save",
                "arguments": {
                    "project_id": self.test_project_id,
                    "context": context_data,
                    "metadata": {
                        "test_run": True,
                        "validator_version": "1.0.0"
                    }
                }
            })
            
            # Load project context
            load_response = await self._send_mcp_request(process, "tools/call", {
                "name": "project_context_load",
                "arguments": {
                    "project_id": self.test_project_id,
                    "include_memories": True
                }
            })
            
            process.terminate()
            
            return TestResult(
                test_name="Project Context Operations",
                passed=True,
                duration_ms=0,
                actual_output={
                    "save": save_response["result"],
                    "load": load_response["result"]
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="Project Context Operations",
                passed=False,
                duration_ms=0,
                error_message=str(e)
            )
    
    async def test_hybrid_search(self) -> TestResult:
        """Test hybrid search functionality"""
        try:
            process = await self._start_mcp_server()
            
            # Initialize
            await self._send_mcp_request(process, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "validator", "version": "1.0.0"}
            })
            
            # Ingest documents with different content
            docs = [
                "Python programming language is excellent for data science and machine learning.",
                "JavaScript is the primary language for web development and frontend applications.",
                "Rust programming offers memory safety and high performance for system programming."
            ]
            
            for i, doc in enumerate(docs):
                await self._send_mcp_request(process, "tools/call", {
                    "name": "ingest_document",
                    "arguments": {
                        "content": doc,
                        "metadata": {"doc_id": f"test_doc_{i}"}
                    }
                })
            
            # Wait for indexing
            await asyncio.sleep(1)
            
            # Perform hybrid search
            response = await self._send_mcp_request(process, "tools/call", {
                "name": "hybrid_search",
                "arguments": {
                    "query": "programming languages for web",
                    "limit": 3,
                    "keyword_weight": 0.3
                }
            })
            
            process.terminate()
            
            return TestResult(
                test_name="Hybrid Search",
                passed=True,
                duration_ms=0,
                actual_output=response["result"]
            )
            
        except Exception as e:
            return TestResult(
                test_name="Hybrid Search",
                passed=False,
                duration_ms=0,
                error_message=str(e)
            )
    
    async def test_error_handling(self) -> TestResult:
        """Test error handling for invalid requests"""
        try:
            process = await self._start_mcp_server()
            
            # Initialize
            await self._send_mcp_request(process, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "validator", "version": "1.0.0"}
            })
            
            # Test invalid tool name
            response1 = await self._send_mcp_request(process, "tools/call", {
                "name": "nonexistent_tool",
                "arguments": {}
            })
            
            # Test invalid parameters
            response2 = await self._send_mcp_request(process, "tools/call", {
                "name": "search",
                "arguments": {
                    "invalid_param": "test"
                }
            })
            
            process.terminate()
            
            # Check if errors are handled gracefully
            errors_handled = (
                "error" in response1 and
                "error" in response2
            )
            
            return TestResult(
                test_name="Error Handling",
                passed=errors_handled,
                duration_ms=0,
                actual_output={
                    "invalid_tool": response1,
                    "invalid_params": response2
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="Error Handling",
                passed=False,
                duration_ms=0,
                error_message=str(e)
            )
    
    async def test_performance_benchmarks(self) -> TestResult:
        """Run performance benchmarks"""
        try:
            process = await self._start_mcp_server()
            
            # Initialize
            await self._send_mcp_request(process, "initialize", {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "validator", "version": "1.0.0"}
            })
            
            benchmarks = {}
            
            # Document ingestion benchmark
            ingest_times = []
            for i in range(5):
                start = time.time()
                await self._send_mcp_request(process, "tools/call", {
                    "name": "ingest_document",
                    "arguments": {
                        "content": f"Benchmark document {i} with content about various topics including technology, science, and programming.",
                        "metadata": {"benchmark": True, "doc_id": i}
                    }
                })
                ingest_times.append((time.time() - start) * 1000)
            
            benchmarks["avg_ingestion_ms"] = statistics.mean(ingest_times)
            
            # Search benchmark
            search_times = []
            for i in range(10):
                start = time.time()
                await self._send_mcp_request(process, "tools/call", {
                    "name": "search",
                    "arguments": {
                        "query": f"technology benchmark query {i}",
                        "limit": 5
                    }
                })
                search_times.append((time.time() - start) * 1000)
            
            benchmarks["avg_search_ms"] = statistics.mean(search_times)
            
            # Memory operations benchmark
            memory_times = []
            for i in range(5):
                start = time.time()
                await self._send_mcp_request(process, "tools/call", {
                    "name": "memory_store",
                    "arguments": {
                        "content": f"Benchmark memory item {i}",
                        "memory_type": "working",
                        "importance": 0.5
                    }
                })
                memory_times.append((time.time() - start) * 1000)
            
            benchmarks["avg_memory_store_ms"] = statistics.mean(memory_times)
            
            process.terminate()
            
            return TestResult(
                test_name="Performance Benchmarks",
                passed=True,
                duration_ms=0,
                performance_metrics=benchmarks
            )
            
        except Exception as e:
            return TestResult(
                test_name="Performance Benchmarks",
                passed=False,
                duration_ms=0,
                error_message=str(e)
            )
    
    async def test_claude_cli_integration(self) -> TestResult:
        """Test Claude CLI integration"""
        try:
            # Check if Claude CLI is available
            result = subprocess.run(
                ["claude", "--version"],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode != 0:
                raise Exception("Claude CLI not available")
            
            # Test MCP configuration with Claude CLI
            test_cmd = [
                "claude",
                "--strict-mcp-config",
                "--mcp-config", str(self.mcp_config_path),
                "--debug",
                "Test RAG-Redis MCP integration"
            ]
            
            result = subprocess.run(
                test_cmd,
                capture_output=True,
                text=True,
                timeout=30
            )
            
            # Check if command executed without critical errors
            success = result.returncode == 0 or "error" not in result.stderr.lower()
            
            return TestResult(
                test_name="Claude CLI Integration",
                passed=success,
                duration_ms=0,
                actual_output={
                    "returncode": result.returncode,
                    "stdout": result.stdout[:500],  # Truncate for readability
                    "stderr": result.stderr[:500]
                }
            )
            
        except Exception as e:
            return TestResult(
                test_name="Claude CLI Integration",
                passed=False,
                duration_ms=0,
                error_message=str(e)
            )
    
    # ============================================================================
    # MAIN VALIDATION METHODS
    # ============================================================================
    
    async def run_all_tests(self):
        """Run all validation tests"""
        self._print_header("RAG-Redis MCP Validation Suite")
        
        print(f"{Colors.YELLOW}Project Root: {self.project_root}{Colors.END}")
        print(f"{Colors.YELLOW}MCP Config: {self.mcp_config_path}{Colors.END}")
        print(f"{Colors.YELLOW}Server Binary: {self.server_binary}{Colors.END}")
        print(f"{Colors.YELLOW}Redis URL: {self.redis_url}{Colors.END}")
        
        # Pre-flight checks
        self._print_header("Pre-flight Checks")
        await self._run_test("MCP Config Validation", self.test_mcp_config_validation)
        await self._run_test("Redis Connectivity", self.test_redis_connectivity)
        await self._run_test("Server Binary Exists", self.test_server_binary_exists)
        
        # Core functionality tests
        self._print_header("Core Functionality Tests")
        await self._run_test("Server Startup", self.test_server_startup)
        await self._run_test("Health Check Tool", self.test_health_check_tool)
        await self._run_test("Document Ingestion", self.test_document_ingestion)
        await self._run_test("Semantic Search", self.test_semantic_search)
        await self._run_test("Hybrid Search", self.test_hybrid_search)
        
        # Memory system tests
        self._print_header("Memory System Tests")
        await self._run_test("Memory Operations", self.test_memory_operations)
        await self._run_test("Agent Memory Integration", self.test_agent_memory_integration)
        await self._run_test("Project Context Operations", self.test_project_context_operations)
        
        # Error handling and edge cases
        self._print_header("Error Handling Tests")
        await self._run_test("Error Handling", self.test_error_handling)
        
        # Performance benchmarks
        self._print_header("Performance Benchmarks")
        await self._run_test("Performance Benchmarks", self.test_performance_benchmarks)
        
        # Integration tests
        self._print_header("Integration Tests")
        await self._run_test("Claude CLI Integration", self.test_claude_cli_integration)
    
    def generate_summary(self) -> ValidationSummary:
        """Generate validation summary"""
        passed_tests = [r for r in self.results if r.passed]
        failed_tests = [r for r in self.results if not r.passed]
        
        total_duration = sum(r.duration_ms for r in self.results) / 1000
        avg_duration = statistics.mean([r.duration_ms for r in self.results]) if self.results else 0
        
        # Collect performance metrics
        performance_benchmarks = {}
        for result in self.results:
            if result.performance_metrics:
                performance_benchmarks.update(result.performance_metrics)
        
        # Identify critical failures
        critical_failures = []
        critical_tests = ["Redis Connectivity", "Server Binary Exists", "Server Startup", "Health Check Tool"]
        for result in failed_tests:
            if result.test_name in critical_tests:
                critical_failures.append(result.test_name)
        
        # Generate warnings
        warnings = []
        if len(failed_tests) > 0:
            warnings.append(f"{len(failed_tests)} tests failed")
        
        performance_benchmarks_results = [r for r in self.results if r.performance_metrics]
        if performance_benchmarks_results:
            perf_result = performance_benchmarks_results[0]
            if perf_result.performance_metrics.get("avg_search_ms", 0) > 100:
                warnings.append("Search performance slower than expected")
            if perf_result.performance_metrics.get("avg_ingestion_ms", 0) > 500:
                warnings.append("Ingestion performance slower than expected")
        
        return ValidationSummary(
            total_tests=len(self.results),
            passed_tests=len(passed_tests),
            failed_tests=len(failed_tests),
            total_duration_s=total_duration,
            average_test_time_ms=avg_duration,
            performance_benchmarks=performance_benchmarks,
            critical_failures=critical_failures,
            warnings=warnings
        )
    
    def print_summary(self, summary: ValidationSummary):
        """Print validation summary"""
        self._print_header("Validation Summary")
        
        # Overall results
        print(f"{Colors.BOLD}Test Results:{Colors.END}")
        print(f"  Total Tests: {summary.total_tests}")
        print(f"  {Colors.GREEN}Passed: {summary.passed_tests}{Colors.END}")
        print(f"  {Colors.RED}Failed: {summary.failed_tests}{Colors.END}")
        print(f"  Success Rate: {(summary.passed_tests / summary.total_tests * 100):.1f}%")
        print(f"  Total Duration: {summary.total_duration_s:.2f}s")
        print(f"  Average Test Time: {summary.average_test_time_ms:.1f}ms")
        
        # Performance benchmarks
        if summary.performance_benchmarks:
            print(f"\n{Colors.BOLD}Performance Benchmarks:{Colors.END}")
            for metric, value in summary.performance_benchmarks.items():
                print(f"  {metric}: {value:.2f}")
        
        # Critical failures
        if summary.critical_failures:
            print(f"\n{Colors.BOLD}{Colors.RED}Critical Failures:{Colors.END}")
            for failure in summary.critical_failures:
                print(f"  {Colors.RED}- {failure}{Colors.END}")
        
        # Warnings
        if summary.warnings:
            print(f"\n{Colors.BOLD}{Colors.YELLOW}Warnings:{Colors.END}")
            for warning in summary.warnings:
                print(f"  {Colors.YELLOW}- {warning}{Colors.END}")
        
        # Failed tests details
        failed_tests = [r for r in self.results if not r.passed]
        if failed_tests:
            print(f"\n{Colors.BOLD}{Colors.RED}Failed Tests Details:{Colors.END}")
            for result in failed_tests:
                print(f"  {Colors.RED}- {result.test_name}{Colors.END}")
                if result.error_message:
                    print(f"    Error: {result.error_message}")
        
        # Overall status
        print(f"\n{Colors.BOLD}Overall Status:{Colors.END}")
        if summary.failed_tests == 0:
            print(f"  {Colors.GREEN}[OK] ALL TESTS PASSED - MCP Server is fully functional{Colors.END}")
        elif len(summary.critical_failures) == 0:
            print(f"  {Colors.YELLOW}[WARN] SOME TESTS FAILED - Core functionality works but issues detected{Colors.END}")
        else:
            print(f"  {Colors.RED}[FAIL] CRITICAL FAILURES - MCP Server has serious issues{Colors.END}")
    
    def save_results(self, summary: ValidationSummary):
        """Save detailed results to file"""
        results_data = {
            "timestamp": datetime.now().isoformat(),
            "summary": asdict(summary),
            "detailed_results": [asdict(r) for r in self.results],
            "configuration": {
                "project_root": str(self.project_root),
                "mcp_config_path": str(self.mcp_config_path),
                "server_binary": str(self.server_binary),
                "redis_url": self.redis_url
            }
        }
        
        results_file = self.project_root / "mcp_validation_results.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, default=str, ensure_ascii=False)
        
        print(f"\n{Colors.CYAN}Detailed results saved to: {results_file}{Colors.END}")

async def main():
    """Main validation function"""
    if len(sys.argv) != 2:
        print(f"Usage: {sys.argv[0]} <project_root>")
        sys.exit(1)
    
    project_root = sys.argv[1]
    validator = MCPValidator(project_root)
    
    try:
        await validator.run_all_tests()
        summary = validator.generate_summary()
        validator.print_summary(summary)
        validator.save_results(summary)
        
        # Exit with error code if critical failures
        if summary.critical_failures:
            sys.exit(1)
        elif summary.failed_tests > 0:
            sys.exit(2)  # Non-critical failures
        else:
            sys.exit(0)  # All tests passed
            
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Validation interrupted by user{Colors.END}")
        sys.exit(130)
    except Exception as e:
        print(f"\n{Colors.RED}Validation failed with error: {e}{Colors.END}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())