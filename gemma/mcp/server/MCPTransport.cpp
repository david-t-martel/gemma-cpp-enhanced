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

#include "MCPTransport.h"
#include <sstream>
#include <regex>
#include <algorithm>
#include <chrono>
#include <cstring>

#ifdef _WIN32
#include <winsock2.h>
#include <ws2tcpip.h>
#pragma comment(lib, "ws2_32.lib")
#else
#include <sys/socket.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <netdb.h>
#include <unistd.h>
#include <fcntl.h>
#endif

namespace gcpp {
namespace mcp {

namespace {

// Platform-specific socket initialization
class SocketInitializer {
public:
  SocketInitializer() {
#ifdef _WIN32
    WSADATA wsa_data;
    WSAStartup(MAKEWORD(2, 2), &wsa_data);
#endif
  }
  
  ~SocketInitializer() {
#ifdef _WIN32
    WSACleanup();
#endif
  }
};

static SocketInitializer socket_initializer;

// Cross-platform socket close
void CloseSocketFd(int fd) {
#ifdef _WIN32
  closesocket(fd);
#else
  close(fd);
#endif
}

// Cross-platform socket error check
bool IsSocketError(int result) {
#ifdef _WIN32
  return result == SOCKET_ERROR;
#else
  return result < 0;
#endif
}

// Get last socket error
int GetLastSocketError() {
#ifdef _WIN32
  return WSAGetLastError();
#else
  return errno;
#endif
}

}  // namespace

// StdioTransport implementation
StdioTransport::StdioTransport() = default;

StdioTransport::~StdioTransport() {
  Stop();
}

bool StdioTransport::Start() {
  if (running_.load()) {
    return true;  // Already running
  }

  status_.store(TransportStatus::CONNECTING);
  running_.store(true);

  try {
    read_thread_ = std::thread(&StdioTransport::ReadLoop, this);
    write_thread_ = std::thread(&StdioTransport::WriteLoop, this);
    
    NotifyStatus(TransportStatus::CONNECTED, "Stdio transport started");
    return true;
  } catch (const std::exception& e) {
    running_.store(false);
    NotifyStatus(TransportStatus::ERROR, std::string("Failed to start stdio transport: ") + e.what());
    return false;
  }
}

void StdioTransport::Stop() {
  if (!running_.load()) {
    return;
  }

  running_.store(false);
  status_.store(TransportStatus::DISCONNECTING);

  // Wake up write thread
  {
    std::lock_guard<std::mutex> lock(write_mutex_);
    write_cv_.notify_all();
  }

  if (read_thread_.joinable()) {
    read_thread_.join();
  }
  if (write_thread_.joinable()) {
    write_thread_.join();
  }

  NotifyStatus(TransportStatus::DISCONNECTED, "Stdio transport stopped");
}

bool StdioTransport::SendMessage(const std::string& message) {
  if (!IsConnected()) {
    return false;
  }

  {
    std::lock_guard<std::mutex> lock(write_mutex_);
    write_queue_.push(message);
    write_cv_.notify_one();
  }

  return true;
}

bool StdioTransport::IsConnected() const {
  return status_.load() == TransportStatus::CONNECTED;
}

TransportStatus StdioTransport::GetStatus() const {
  return status_.load();
}

void StdioTransport::SetMessageCallback(MessageCallback callback) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  message_callback_ = callback;
}

void StdioTransport::SetStatusCallback(StatusCallback callback) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  status_callback_ = callback;
}

std::string StdioTransport::GetInfo() const {
  return "Standard input/output transport";
}

void StdioTransport::ReadLoop() {
  std::string line;
  while (running_.load()) {
    if (std::getline(std::cin, line)) {
      if (!line.empty()) {
        std::lock_guard<std::mutex> lock(callback_mutex_);
        if (message_callback_) {
          message_callback_(line);
        }
      }
    } else {
      // EOF or error
      if (running_.load()) {
        NotifyStatus(TransportStatus::ERROR, "EOF on stdin");
      }
      break;
    }
  }
}

void StdioTransport::WriteLoop() {
  while (running_.load()) {
    std::unique_lock<std::mutex> lock(write_mutex_);
    write_cv_.wait(lock, [this] { return !write_queue_.empty() || !running_.load(); });

    while (!write_queue_.empty() && running_.load()) {
      std::string message = write_queue_.front();
      write_queue_.pop();
      lock.unlock();

      std::cout << message << std::endl;
      std::cout.flush();

      lock.lock();
    }
  }
}

void StdioTransport::NotifyStatus(TransportStatus status, const std::string& details) {
  status_.store(status);
  std::lock_guard<std::mutex> lock(callback_mutex_);
  if (status_callback_) {
    status_callback_(status, details);
  }
}

// TcpTransport implementation
TcpTransport::TcpTransport(const std::string& host, int port) 
    : host_(host), port_(port) {}

TcpTransport::~TcpTransport() {
  Stop();
}

bool TcpTransport::Start() {
  if (running_.load()) {
    return true;
  }

  status_.store(TransportStatus::CONNECTING);
  
  if (!CreateSocket()) {
    NotifyStatus(TransportStatus::ERROR, "Failed to create socket");
    return false;
  }

  running_.store(true);

  try {
    read_thread_ = std::thread(&TcpTransport::ReadLoop, this);
    write_thread_ = std::thread(&TcpTransport::WriteLoop, this);
    
    NotifyStatus(TransportStatus::CONNECTED, "TCP connection established");
    return true;
  } catch (const std::exception& e) {
    running_.store(false);
    CloseSocket();
    NotifyStatus(TransportStatus::ERROR, std::string("Failed to start TCP transport: ") + e.what());
    return false;
  }
}

void TcpTransport::Stop() {
  if (!running_.load()) {
    return;
  }

  running_.store(false);
  status_.store(TransportStatus::DISCONNECTING);

  CloseSocket();

  {
    std::lock_guard<std::mutex> lock(write_mutex_);
    write_cv_.notify_all();
  }

  if (read_thread_.joinable()) {
    read_thread_.join();
  }
  if (write_thread_.joinable()) {
    write_thread_.join();
  }

  NotifyStatus(TransportStatus::DISCONNECTED, "TCP transport stopped");
}

bool TcpTransport::SendMessage(const std::string& message) {
  if (!IsConnected()) {
    return false;
  }

  std::string prepared = PrepareMessage(message);
  
  {
    std::lock_guard<std::mutex> lock(write_mutex_);
    write_queue_.push(prepared);
    write_cv_.notify_one();
  }

  return true;
}

bool TcpTransport::IsConnected() const {
  std::lock_guard<std::mutex> lock(socket_mutex_);
  return status_.load() == TransportStatus::CONNECTED && socket_fd_ != -1;
}

TransportStatus TcpTransport::GetStatus() const {
  return status_.load();
}

void TcpTransport::SetMessageCallback(MessageCallback callback) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  message_callback_ = callback;
}

void TcpTransport::SetStatusCallback(StatusCallback callback) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  status_callback_ = callback;
}

std::string TcpTransport::GetInfo() const {
  return "TCP connection to " + host_ + ":" + std::to_string(port_);
}

std::string TcpTransport::PrepareMessage(const std::string& message) {
  // Default: just add newline for line-based protocol
  return message + "\n";
}

std::string TcpTransport::ProcessReceived(const std::string& data) {
  // Default: return as-is
  return data;
}

bool TcpTransport::CreateSocket() {
  std::lock_guard<std::mutex> lock(socket_mutex_);

  socket_fd_ = socket(AF_INET, SOCK_STREAM, 0);
  if (socket_fd_ == -1) {
    return false;
  }

  // Resolve hostname
  struct hostent* host_entry = gethostbyname(host_.c_str());
  if (!host_entry) {
    CloseSocketFd(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  // Setup address
  struct sockaddr_in server_addr;
  memset(&server_addr, 0, sizeof(server_addr));
  server_addr.sin_family = AF_INET;
  server_addr.sin_port = htons(port_);
  memcpy(&server_addr.sin_addr.s_addr, host_entry->h_addr, host_entry->h_length);

  // Connect
  if (connect(socket_fd_, (struct sockaddr*)&server_addr, sizeof(server_addr)) == -1) {
    CloseSocketFd(socket_fd_);
    socket_fd_ = -1;
    return false;
  }

  return true;
}

void TcpTransport::CloseSocket() {
  std::lock_guard<std::mutex> lock(socket_mutex_);
  if (socket_fd_ != -1) {
    CloseSocketFd(socket_fd_);
    socket_fd_ = -1;
  }
}

void TcpTransport::ReadLoop() {
  char buffer[4096];
  
  while (running_.load()) {
    std::lock_guard<std::mutex> lock(socket_mutex_);
    if (socket_fd_ == -1) {
      break;
    }

    int bytes_received = recv(socket_fd_, buffer, sizeof(buffer) - 1, 0);
    
    if (bytes_received > 0) {
      buffer[bytes_received] = '\0';
      receive_buffer_ += buffer;
      
      // Process complete messages (assuming line-based for now)
      std::string processed = ProcessReceived(receive_buffer_);
      if (!processed.empty()) {
        receive_buffer_.clear();
        
        std::lock_guard<std::mutex> cb_lock(callback_mutex_);
        if (message_callback_) {
          message_callback_(processed);
        }
      }
    } else if (bytes_received == 0) {
      // Connection closed
      if (running_.load()) {
        NotifyStatus(TransportStatus::ERROR, "Connection closed by peer");
      }
      break;
    } else {
      // Error
      if (running_.load()) {
        NotifyStatus(TransportStatus::ERROR, "Socket read error: " + std::to_string(GetLastSocketError()));
      }
      break;
    }
  }
}

void TcpTransport::WriteLoop() {
  while (running_.load()) {
    std::unique_lock<std::mutex> lock(write_mutex_);
    write_cv_.wait(lock, [this] { return !write_queue_.empty() || !running_.load(); });

    while (!write_queue_.empty() && running_.load()) {
      std::string message = write_queue_.front();
      write_queue_.pop();
      lock.unlock();

      {
        std::lock_guard<std::mutex> socket_lock(socket_mutex_);
        if (socket_fd_ != -1) {
          int bytes_sent = send(socket_fd_, message.c_str(), message.length(), 0);
          if (IsSocketError(bytes_sent)) {
            NotifyStatus(TransportStatus::ERROR, "Socket write error");
            break;
          }
        }
      }

      lock.lock();
    }
  }
}

void TcpTransport::NotifyStatus(TransportStatus status, const std::string& details) {
  status_.store(status);
  std::lock_guard<std::mutex> lock(callback_mutex_);
  if (status_callback_) {
    status_callback_(status, details);
  }
}

// WebSocketTransport implementation
WebSocketTransport::WebSocketTransport(const std::string& host, int port, const std::string& path)
    : TcpTransport(host, port), path_(path) {}

WebSocketTransport::~WebSocketTransport() = default;

std::string WebSocketTransport::GetInfo() const {
  return "WebSocket connection to ws://" + host_ + ":" + std::to_string(port_) + path_;
}

std::string WebSocketTransport::PrepareMessage(const std::string& message) {
  if (!handshake_complete_) {
    return message;  // During handshake, send raw
  }
  return CreateWebSocketFrame(message);
}

std::string WebSocketTransport::ProcessReceived(const std::string& data) {
  if (!handshake_complete_) {
    handshake_buffer_ += data;
    
    // Look for end of HTTP headers
    auto header_end = handshake_buffer_.find("\r\n\r\n");
    if (header_end != std::string::npos) {
      // Check if handshake response is valid
      if (handshake_buffer_.find("HTTP/1.1 101") != std::string::npos &&
          handshake_buffer_.find("Upgrade: websocket") != std::string::npos) {
        handshake_complete_ = true;
        handshake_buffer_.clear();
        return "";  // Handshake complete, no message to process
      } else {
        NotifyStatus(TransportStatus::ERROR, "WebSocket handshake failed");
        return "";
      }
    }
    return "";  // Still waiting for complete handshake
  }
  
  return ParseWebSocketFrame(data);
}

bool WebSocketTransport::CreateSocket() {
  if (!TcpTransport::CreateSocket()) {
    return false;
  }
  
  return PerformHandshake();
}

bool WebSocketTransport::PerformHandshake() {
  // Simple WebSocket handshake (production code should generate proper key)
  std::string handshake = 
    "GET " + path_ + " HTTP/1.1\r\n"
    "Host: " + host_ + ":" + std::to_string(port_) + "\r\n"
    "Upgrade: websocket\r\n"
    "Connection: Upgrade\r\n"
    "Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==\r\n"
    "Sec-WebSocket-Version: 13\r\n"
    "\r\n";

  {
    std::lock_guard<std::mutex> lock(socket_mutex_);
    if (socket_fd_ != -1) {
      int bytes_sent = send(socket_fd_, handshake.c_str(), handshake.length(), 0);
      return !IsSocketError(bytes_sent);
    }
  }
  
  return false;
}

std::string WebSocketTransport::CreateWebSocketFrame(const std::string& payload) {
  // Simple text frame (production code should handle all frame types)
  std::string frame;
  
  // FIN bit set, opcode for text frame
  frame.push_back(0x81);
  
  size_t payload_len = payload.length();
  if (payload_len < 126) {
    frame.push_back(static_cast<char>(payload_len | 0x80));  // Masked
  } else if (payload_len < 65536) {
    frame.push_back(0xFE);  // 126 + masked
    frame.push_back(static_cast<char>((payload_len >> 8) & 0xFF));
    frame.push_back(static_cast<char>(payload_len & 0xFF));
  } else {
    // 64-bit length (simplified)
    frame.push_back(0xFF);  // 127 + masked
    for (int i = 7; i >= 0; --i) {
      frame.push_back(static_cast<char>((payload_len >> (i * 8)) & 0xFF));
    }
  }
  
  // Simple masking key (production should use random)
  char mask[4] = {0x12, 0x34, 0x56, 0x78};
  frame.append(mask, 4);
  
  // Masked payload
  for (size_t i = 0; i < payload_len; ++i) {
    frame.push_back(payload[i] ^ mask[i % 4]);
  }
  
  return frame;
}

std::string WebSocketTransport::ParseWebSocketFrame(const std::string& frame) {
  // Simplified WebSocket frame parsing
  if (frame.length() < 2) {
    return "";
  }
  
  unsigned char opcode = frame[0] & 0x0F;
  if (opcode != 0x01) {  // Text frame
    return "";
  }
  
  size_t payload_start = 2;
  unsigned char payload_len = frame[1] & 0x7F;
  
  if (payload_len == 126) {
    if (frame.length() < 4) return "";
    payload_len = (static_cast<unsigned char>(frame[2]) << 8) | 
                  static_cast<unsigned char>(frame[3]);
    payload_start = 4;
  } else if (payload_len == 127) {
    payload_start = 10;  // Skip 8-byte length
  }
  
  if (frame.length() < payload_start + payload_len) {
    return "";
  }
  
  return frame.substr(payload_start, payload_len);
}

// TransportFactory implementation
std::unique_ptr<MCPTransport> TransportFactory::CreateTransport(const std::string& uri) {
  std::string scheme, host, path;
  int port;
  
  if (!ParseUri(uri, scheme, host, port, path)) {
    return nullptr;
  }
  
  if (scheme == "stdio") {
    return CreateStdioTransport();
  } else if (scheme == "tcp") {
    return CreateTcpTransport(host, port);
  } else if (scheme == "ws" || scheme == "websocket") {
    return CreateWebSocketTransport(host, port, path);
  }
  
  return nullptr;
}

std::unique_ptr<MCPTransport> TransportFactory::CreateStdioTransport() {
  return std::make_unique<StdioTransport>();
}

std::unique_ptr<MCPTransport> TransportFactory::CreateTcpTransport(const std::string& host, int port) {
  return std::make_unique<TcpTransport>(host, port);
}

std::unique_ptr<MCPTransport> TransportFactory::CreateWebSocketTransport(const std::string& host, int port, 
                                                                         const std::string& path) {
  return std::make_unique<WebSocketTransport>(host, port, path);
}

bool TransportFactory::ParseUri(const std::string& uri, std::string& scheme, 
                               std::string& host, int& port, std::string& path) {
  // Simple URI parsing (production should use proper URI parser)
  std::regex uri_regex(R"(^([a-zA-Z][a-zA-Z0-9+.-]*):(?://([^:/]+)(?::(\d+))?(/.*)?)?$)");
  std::smatch matches;
  
  if (!std::regex_match(uri, matches, uri_regex)) {
    return false;
  }
  
  scheme = matches[1].str();
  host = matches[2].str();
  
  if (matches[3].matched) {
    port = std::stoi(matches[3].str());
  } else {
    port = transport_utils::GetDefaultPort(scheme);
  }
  
  path = matches[4].matched ? matches[4].str() : "/";
  
  return true;
}

// ConnectionManager implementation
ConnectionManager::ConnectionManager() = default;

ConnectionManager::~ConnectionManager() {
  StopAll();
}

bool ConnectionManager::AddTransport(std::unique_ptr<MCPTransport> transport, const std::string& name) {
  if (!transport) {
    return false;
  }
  
  std::lock_guard<std::mutex> lock(transports_mutex_);
  
  std::string transport_name = name.empty() ? 
    transport->GetType() + "_" + std::to_string(transports_.size()) : name;
  
  // Set up callbacks
  transport->SetMessageCallback([this, transport_name](const std::string& message) {
    OnTransportMessage(transport_name, message);
  });
  
  transport->SetStatusCallback([this, transport_name](TransportStatus status, const std::string& details) {
    OnTransportStatus(transport_name, status, details);
  });
  
  transports_.push_back({std::move(transport), transport_name});
  return true;
}

bool ConnectionManager::StartAll() {
  std::lock_guard<std::mutex> lock(transports_mutex_);
  
  bool all_started = true;
  for (auto& info : transports_) {
    if (!info.transport->Start()) {
      all_started = false;
    }
  }
  
  return all_started;
}

void ConnectionManager::StopAll() {
  std::lock_guard<std::mutex> lock(transports_mutex_);
  
  for (auto& info : transports_) {
    info.transport->Stop();
  }
}

bool ConnectionManager::SendMessage(const std::string& message, const std::string& transport_name) {
  std::lock_guard<std::mutex> lock(transports_mutex_);
  
  if (transport_name.empty()) {
    // Send to first connected transport
    for (auto& info : transports_) {
      if (info.transport->IsConnected()) {
        return info.transport->SendMessage(message);
      }
    }
    return false;
  } else {
    // Send to specific transport
    auto it = std::find_if(transports_.begin(), transports_.end(),
                          [&transport_name](const TransportInfo& info) {
                            return info.name == transport_name;
                          });
    if (it != transports_.end() && it->transport->IsConnected()) {
      return it->transport->SendMessage(message);
    }
    return false;
  }
}

void ConnectionManager::BroadcastMessage(const std::string& message) {
  std::lock_guard<std::mutex> lock(transports_mutex_);
  
  for (auto& info : transports_) {
    if (info.transport->IsConnected()) {
      info.transport->SendMessage(message);
    }
  }
}

void ConnectionManager::SetMessageCallback(MessageCallback callback) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  global_message_callback_ = callback;
}

void ConnectionManager::SetStatusCallback(StatusCallback callback) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  global_status_callback_ = callback;
}

MCPTransport* ConnectionManager::GetTransport(const std::string& name) const {
  std::lock_guard<std::mutex> lock(transports_mutex_);
  
  auto it = std::find_if(transports_.begin(), transports_.end(),
                        [&name](const TransportInfo& info) {
                          return info.name == name;
                        });
  return (it != transports_.end()) ? it->transport.get() : nullptr;
}

std::vector<std::string> ConnectionManager::GetTransportNames() const {
  std::lock_guard<std::mutex> lock(transports_mutex_);
  
  std::vector<std::string> names;
  for (const auto& info : transports_) {
    names.push_back(info.name);
  }
  return names;
}

std::map<std::string, TransportStatus> ConnectionManager::GetAllStatuses() const {
  std::lock_guard<std::mutex> lock(transports_mutex_);
  
  std::map<std::string, TransportStatus> statuses;
  for (const auto& info : transports_) {
    statuses[info.name] = info.transport->GetStatus();
  }
  return statuses;
}

void ConnectionManager::OnTransportMessage(const std::string& transport_name, const std::string& message) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  if (global_message_callback_) {
    global_message_callback_(message);
  }
}

void ConnectionManager::OnTransportStatus(const std::string& transport_name, TransportStatus status, 
                                         const std::string& details) {
  std::lock_guard<std::mutex> lock(callback_mutex_);
  if (global_status_callback_) {
    global_status_callback_(status, transport_name + ": " + details);
  }
}

// transport_utils implementation
namespace transport_utils {

bool IsSupportedScheme(const std::string& scheme) {
  static const std::vector<std::string> supported = {"stdio", "tcp", "ws", "websocket"};
  return std::find(supported.begin(), supported.end(), scheme) != supported.end();
}

int GetDefaultPort(const std::string& scheme) {
  if (scheme == "tcp") return 8080;
  if (scheme == "ws" || scheme == "websocket") return 8080;
  return -1;  // No default port
}

bool ValidateUri(const std::string& uri) {
  std::string scheme, host, path;
  int port;
  return TransportFactory::ParseUri(uri, scheme, host, port, path) && 
         IsSupportedScheme(scheme);
}

std::string CreateUri(const std::string& scheme, const std::string& host, 
                     int port, const std::string& path) {
  std::ostringstream oss;
  oss << scheme << ":";
  
  if (!host.empty()) {
    oss << "//" << host;
    if (port != GetDefaultPort(scheme) && port > 0) {
      oss << ":" << port;
    }
    if (!path.empty() && path != "/") {
      oss << path;
    }
  }
  
  return oss.str();
}

std::vector<std::string> GetAvailableTransportTypes() {
  return {"stdio", "tcp", "websocket"};
}

}  // namespace transport_utils

}  // namespace mcp
}  // namespace gcpp