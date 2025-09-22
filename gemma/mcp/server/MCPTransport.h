// Copyright 2024 Google LLC
// SPDX-License-Identifier: Apache-2.0
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_GEMMA_CPP_INTERFACES_MCP_MCPTRANSPORT_H_
#define THIRD_PARTY_GEMMA_CPP_INTERFACES_MCP_MCPTRANSPORT_H_

#include <string>
#include <memory>
#include <functional>
#include <future>
#include <thread>
#include <atomic>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <iostream>

namespace gcpp {
namespace mcp {

// Forward declarations
class MCPServer;

// Transport status enumeration
enum class TransportStatus {
  DISCONNECTED,
  CONNECTING,
  CONNECTED,
  DISCONNECTING,
  ERROR
};

// Message callback type - called when a message is received
using MessageCallback = std::function<void(const std::string& message)>;

// Connection status callback type
using StatusCallback = std::function<void(TransportStatus status, const std::string& details)>;

// Abstract base class for MCP transport layers
class MCPTransport {
public:
  virtual ~MCPTransport() = default;

  // Start the transport (begin listening/connecting)
  virtual bool Start() = 0;

  // Stop the transport
  virtual void Stop() = 0;

  // Send a message through the transport
  virtual bool SendMessage(const std::string& message) = 0;

  // Check if transport is connected and ready
  virtual bool IsConnected() const = 0;

  // Get current transport status
  virtual TransportStatus GetStatus() const = 0;

  // Set callback for received messages
  virtual void SetMessageCallback(MessageCallback callback) = 0;

  // Set callback for status changes
  virtual void SetStatusCallback(StatusCallback callback) = 0;

  // Get transport type identifier
  virtual std::string GetType() const = 0;

  // Get transport-specific information
  virtual std::string GetInfo() const = 0;
};

// Stdio transport implementation for process communication
class StdioTransport : public MCPTransport {
public:
  StdioTransport();
  ~StdioTransport() override;

  // MCPTransport interface
  bool Start() override;
  void Stop() override;
  bool SendMessage(const std::string& message) override;
  bool IsConnected() const override;
  TransportStatus GetStatus() const override;
  void SetMessageCallback(MessageCallback callback) override;
  void SetStatusCallback(StatusCallback callback) override;
  std::string GetType() const override { return "stdio"; }
  std::string GetInfo() const override;

private:
  void ReadLoop();
  void WriteLoop();
  void NotifyStatus(TransportStatus status, const std::string& details = "");

  std::atomic<TransportStatus> status_{TransportStatus::DISCONNECTED};
  MessageCallback message_callback_;
  StatusCallback status_callback_;
  
  std::thread read_thread_;
  std::thread write_thread_;
  std::atomic<bool> running_{false};
  
  // Write queue for thread-safe message sending
  std::queue<std::string> write_queue_;
  std::mutex write_mutex_;
  std::condition_variable write_cv_;
  
  mutable std::mutex callback_mutex_;
};

// TCP transport implementation (base for WebSocket)
class TcpTransport : public MCPTransport {
public:
  TcpTransport(const std::string& host, int port);
  ~TcpTransport() override;

  // MCPTransport interface
  bool Start() override;
  void Stop() override;
  bool SendMessage(const std::string& message) override;
  bool IsConnected() const override;
  TransportStatus GetStatus() const override;
  void SetMessageCallback(MessageCallback callback) override;
  void SetStatusCallback(StatusCallback callback) override;
  std::string GetType() const override { return "tcp"; }
  std::string GetInfo() const override;

protected:
  // Override for custom message processing (e.g., WebSocket framing)
  virtual std::string PrepareMessage(const std::string& message);
  virtual std::string ProcessReceived(const std::string& data);
  
  // Socket handling
  virtual bool CreateSocket();
  virtual void CloseSocket();
  
  std::string host_;
  int port_;
  int socket_fd_{-1};

private:
  void ReadLoop();
  void WriteLoop();
  void NotifyStatus(TransportStatus status, const std::string& details = "");

  std::atomic<TransportStatus> status_{TransportStatus::DISCONNECTED};
  MessageCallback message_callback_;
  StatusCallback status_callback_;
  
  std::thread read_thread_;
  std::thread write_thread_;
  std::atomic<bool> running_{false};
  
  std::queue<std::string> write_queue_;
  std::mutex write_mutex_;
  std::condition_variable write_cv_;
  
  mutable std::mutex callback_mutex_;
  mutable std::mutex socket_mutex_;
  
  // Buffer for partial message assembly
  std::string receive_buffer_;
};

// WebSocket transport implementation
class WebSocketTransport : public TcpTransport {
public:
  WebSocketTransport(const std::string& host, int port, const std::string& path = "/");
  ~WebSocketTransport() override;

  std::string GetType() const override { return "websocket"; }
  std::string GetInfo() const override;

protected:
  std::string PrepareMessage(const std::string& message) override;
  std::string ProcessReceived(const std::string& data) override;
  bool CreateSocket() override;

private:
  bool PerformHandshake();
  std::string CreateWebSocketFrame(const std::string& payload);
  std::string ParseWebSocketFrame(const std::string& frame);
  
  std::string path_;
  bool handshake_complete_{false};
  std::string handshake_buffer_;
};

// Transport factory for creating transport instances
class TransportFactory {
public:
  // Create transport from URI (e.g., "stdio:", "tcp://localhost:8080", "ws://localhost:8080/mcp")
  static std::unique_ptr<MCPTransport> CreateTransport(const std::string& uri);
  
  // Create specific transport types
  static std::unique_ptr<MCPTransport> CreateStdioTransport();
  static std::unique_ptr<MCPTransport> CreateTcpTransport(const std::string& host, int port);
  static std::unique_ptr<MCPTransport> CreateWebSocketTransport(const std::string& host, int port, 
                                                               const std::string& path = "/");

private:
  static bool ParseUri(const std::string& uri, std::string& scheme, 
                      std::string& host, int& port, std::string& path);
};

// Connection manager for handling multiple transport types
class ConnectionManager {
public:
  ConnectionManager();
  ~ConnectionManager();

  // Add a transport to manage
  bool AddTransport(std::unique_ptr<MCPTransport> transport, const std::string& name = "");

  // Start all transports
  bool StartAll();

  // Stop all transports
  void StopAll();

  // Send message through a specific transport
  bool SendMessage(const std::string& message, const std::string& transport_name = "");

  // Broadcast message to all connected transports
  void BroadcastMessage(const std::string& message);

  // Set global message callback (receives messages from any transport)
  void SetMessageCallback(MessageCallback callback);

  // Set global status callback
  void SetStatusCallback(StatusCallback callback);

  // Get transport by name
  MCPTransport* GetTransport(const std::string& name) const;

  // Get list of transport names
  std::vector<std::string> GetTransportNames() const;

  // Get status of all transports
  std::map<std::string, TransportStatus> GetAllStatuses() const;

private:
  void OnTransportMessage(const std::string& transport_name, const std::string& message);
  void OnTransportStatus(const std::string& transport_name, TransportStatus status, 
                        const std::string& details);

  struct TransportInfo {
    std::unique_ptr<MCPTransport> transport;
    std::string name;
  };

  std::vector<TransportInfo> transports_;
  MessageCallback global_message_callback_;
  StatusCallback global_status_callback_;
  mutable std::mutex transports_mutex_;
  mutable std::mutex callback_mutex_;
};

// Utility functions for transport handling
namespace transport_utils {

// Check if a URI scheme is supported
bool IsSupportedScheme(const std::string& scheme);

// Get default port for a scheme
int GetDefaultPort(const std::string& scheme);

// Validate transport URI format
bool ValidateUri(const std::string& uri);

// Create connection string from components
std::string CreateUri(const std::string& scheme, const std::string& host, 
                     int port, const std::string& path = "");

// Get available transport types
std::vector<std::string> GetAvailableTransportTypes();

}  // namespace transport_utils

}  // namespace mcp
}  // namespace gcpp

#endif  // THIRD_PARTY_GEMMA_CPP_INTERFACES_MCP_MCPTRANSPORT_H_