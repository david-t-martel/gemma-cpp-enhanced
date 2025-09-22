#!/usr/bin/env python3
"""
Example usage of the Gemma MCP Server
Demonstrates how to interact with the MCP server via stdio transport
"""

import json
import subprocess
import sys
from typing import Dict, Any, Optional

class GemmaMCPClient:
    """Simple client for interacting with Gemma MCP Server"""

    def __init__(self, model_path: str, tokenizer_path: Optional[str] = None):
        """Initialize client with model paths"""
        cmd = ["./gemma_mcp_stdio_server", "--model", model_path]
        if tokenizer_path:
            cmd.extend(["--tokenizer", tokenizer_path])

        self.process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=0
        )
        self.request_id = 0

    def _send_request(self, method: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        """Send a JSON-RPC request and get response"""
        self.request_id += 1
        request = {
            "jsonrpc": "2.0",
            "id": str(self.request_id),
            "method": method,
            "params": params or {}
        }

        request_json = json.dumps(request)
        print(f"Sending: {request_json}", file=sys.stderr)

        self.process.stdin.write(request_json + "\n")
        self.process.stdin.flush()

        response_line = self.process.stdout.readline()
        if not response_line:
            raise RuntimeError("No response from server")

        response = json.loads(response_line.strip())
        print(f"Received: {json.dumps(response, indent=2)}", file=sys.stderr)

        if "error" in response:
            raise RuntimeError(f"Server error: {response['error']}")

        return response

    def initialize(self) -> Dict[str, Any]:
        """Initialize the MCP connection"""
        return self._send_request("initialize")

    def list_tools(self) -> Dict[str, Any]:
        """List available tools"""
        return self._send_request("tools/list")

    def generate_text(self, prompt: str, **kwargs) -> str:
        """Generate text using the model"""
        params = {
            "name": "generate_text",
            "arguments": {"prompt": prompt, **kwargs}
        }
        response = self._send_request("tools/call", params)

        # Extract text from the response content
        content = response.get("result", {}).get("content", [])
        if content and isinstance(content[0], dict):
            result_text = content[0].get("text", "")
            try:
                result_data = json.loads(result_text)
                return result_data.get("text", "")
            except json.JSONDecodeError:
                return result_text
        return ""

    def count_tokens(self, text: str, include_details: bool = False) -> Dict[str, Any]:
        """Count tokens in text"""
        params = {
            "name": "count_tokens",
            "arguments": {"text": text, "include_details": include_details}
        }
        response = self._send_request("tools/call", params)

        # Extract token info from response
        content = response.get("result", {}).get("content", [])
        if content and isinstance(content[0], dict):
            result_text = content[0].get("text", "")
            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                return {"error": "Failed to parse token response"}
        return {}

    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        params = {
            "name": "get_model_info",
            "arguments": {}
        }
        response = self._send_request("tools/call", params)

        # Extract model info from response
        content = response.get("result", {}).get("content", [])
        if content and isinstance(content[0], dict):
            result_text = content[0].get("text", "")
            try:
                return json.loads(result_text)
            except json.JSONDecodeError:
                return {"error": "Failed to parse model info"}
        return {}

    def close(self):
        """Close the connection"""
        if self.process:
            self.process.stdin.close()
            self.process.terminate()
            self.process.wait()

def main():
    """Example usage of the Gemma MCP client"""

    # Configure model paths (adjust these for your setup)
    model_path = "/c/codedev/llm/.models/gemma2-2b-it-sfp.sbs"
    tokenizer_path = "/c/codedev/llm/.models/tokenizer.spm"

    print("=== Gemma MCP Server Example ===")

    try:
        # Initialize client
        print("Initializing MCP client...")
        client = GemmaMCPClient(model_path, tokenizer_path)

        # Initialize connection
        init_response = client.initialize()
        print("✓ MCP connection initialized")
        print(f"Server: {init_response.get('result', {}).get('serverInfo', {})}")

        # List available tools
        tools_response = client.list_tools()
        tools = tools_response.get("result", {}).get("tools", [])
        print(f"✓ Available tools: {[tool['name'] for tool in tools]}")

        # Get model information
        print("\n--- Model Information ---")
        model_info = client.get_model_info()
        if "error" not in model_info:
            print(f"Model Name: {model_info.get('name', 'Unknown')}")
            print(f"Architecture: {model_info.get('architecture', 'Unknown')}")
            print(f"Parameters: {model_info.get('parameter_count', 'Unknown')}")
            print(f"Context Length: {model_info.get('context_length', 'Unknown')}")
        else:
            print(f"Error getting model info: {model_info}")

        # Test token counting
        print("\n--- Token Counting ---")
        test_text = "Hello, world! This is a test of the Gemma tokenizer."
        token_info = client.count_tokens(test_text, include_details=True)
        if "error" not in token_info:
            print(f"Text: '{test_text}'")
            print(f"Token Count: {token_info.get('token_count', 'Unknown')}")
            print(f"Tokens: {token_info.get('tokens', [])}")
        else:
            print(f"Error counting tokens: {token_info}")

        # Test text generation
        print("\n--- Text Generation ---")
        prompts = [
            "The future of artificial intelligence is",
            "Once upon a time in a land far away,",
            "The three laws of robotics are"
        ]

        for prompt in prompts:
            print(f"\nPrompt: '{prompt}'")
            try:
                generated = client.generate_text(
                    prompt=prompt,
                    temperature=0.8,
                    max_tokens=50,
                    top_k=40
                )
                print(f"Generated: {generated}")
            except Exception as e:
                print(f"Generation failed: {e}")

        # Test different generation parameters
        print("\n--- Parameter Testing ---")
        base_prompt = "Explain machine learning in simple terms:"

        parameters = [
            {"temperature": 0.1, "max_tokens": 30},  # Conservative
            {"temperature": 0.7, "max_tokens": 30},  # Balanced
            {"temperature": 1.2, "max_tokens": 30},  # Creative
        ]

        for i, params in enumerate(parameters):
            print(f"\nTest {i+1} (temp={params['temperature']}):")
            try:
                result = client.generate_text(prompt=base_prompt, **params)
                print(f"Result: {result}")
            except Exception as e:
                print(f"Failed: {e}")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    finally:
        # Clean up
        try:
            client.close()
        except:
            pass

    print("\n=== Example completed successfully ===")
    return 0

if __name__ == "__main__":
    sys.exit(main())