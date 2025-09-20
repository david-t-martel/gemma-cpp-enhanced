# Gemma Chatbot HTTP Server

A production-ready FastAPI HTTP server for the Gemma LLM chatbot with comprehensive features including OpenAI-compatible API endpoints, real-time chat, streaming responses, and production monitoring.

## 🚀 Features

### Core API Functionality
- **OpenAI-Compatible Endpoints**: Full compatibility with OpenAI's chat completions and completions APIs
- **RESTful Design**: Clean, RESTful API design with proper HTTP status codes and error handling
- **Streaming Support**: Server-Sent Events (SSE) and direct streaming for real-time responses
- **WebSocket Chat**: Real-time bidirectional communication for interactive chat experiences

### Production-Ready Features
- **Rate Limiting**: Configurable rate limiting with sliding window algorithm
- **Authentication**: API key-based authentication middleware
- **CORS Support**: Flexible CORS configuration for cross-origin requests
- **Request/Response Logging**: Comprehensive logging with configurable levels
- **Prometheus Metrics**: Built-in metrics collection for monitoring and observability
- **Health Checks**: Multiple health check endpoints (liveness, readiness, detailed status)
- **Graceful Shutdown**: Proper resource cleanup and connection handling during shutdown

### Scalability & Performance
- **Connection Pooling**: Efficient WebSocket connection management
- **Caching**: Response caching with TTL and size limits
- **Async/Await**: Full asynchronous operation for high concurrency
- **Resource Management**: Proper memory and GPU resource handling
- **Background Tasks**: System metrics collection and cleanup tasks

## 📁 Architecture

```
src/server/
├── main.py              # FastAPI application and lifecycle management
├── state.py             # Global application state management
├── middleware.py        # All middleware components
├── websocket.py         # WebSocket connection management
├── api/
│   ├── chat.py          # Chat completion endpoints
│   ├── models.py        # Model management endpoints
│   ├── health.py        # Health check and monitoring
│   └── schemas.py       # Pydantic models for requests/responses
└── __init__.py          # Package exports
```

## 🛠️ API Endpoints

### Chat Completions (OpenAI Compatible)
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "gemma-2b-it",
  "messages": [
    {"role": "user", "content": "Hello!"}
  ],
  "stream": false,
  "temperature": 0.7,
  "max_tokens": 150
}
```

### Text Completions
```http
POST /v1/completions
Content-Type: application/json

{
  "model": "gemma-2b-it",
  "prompt": "The future of AI is",
  "max_tokens": 100,
  "temperature": 0.8
}
```

### Streaming Chat (SSE)
```http
POST /v1/chat/completions
Content-Type: application/json

{
  "model": "gemma-2b-it",
  "messages": [{"role": "user", "content": "Tell me a story"}],
  "stream": true
}
```

### WebSocket Chat
```javascript
const ws = new WebSocket('ws://localhost:8000/ws');
ws.send(JSON.stringify({
  type: 'chat',
  data: {
    message: 'Hello via WebSocket!',
    stream: true
  }
}));
```

### Model Management
```http
GET /v1/models                    # List available models
GET /v1/models/gemma-2b-it       # Get specific model info
POST /v1/models/gemma-2b-it/load  # Load model
```

### Health & Monitoring
```http
GET /health              # Basic health check
GET /health/ready        # Readiness probe
GET /health/live         # Liveness probe
GET /health/status       # Detailed status
GET /health/metrics      # Performance metrics
GET /metrics             # Prometheus metrics
```

## 🚦 Usage

### Starting the Server

#### Option 1: Direct Python Module
```bash
cd /path/to/project
uv run python -m src.server.main
```

#### Option 2: Using uvicorn
```bash
uvicorn src.server.main:app --host 0.0.0.0 --port 8000 --reload
```

#### Option 3: Project Entry Point
```bash
gemma-server  # If installed via setup.py
```

### Configuration

The server uses environment variables with the `GEMMA_` prefix:

```bash
# Server configuration
export GEMMA_SERVER_HOST="0.0.0.0"
export GEMMA_SERVER_PORT="8000"
export GEMMA_SERVER_WORKERS="1"

# Model configuration
export GEMMA_MODEL_NAME="google/gemma-2b-it"
export GEMMA_MODEL_MAX_LENGTH="2048"
export GEMMA_MODEL_TEMPERATURE="0.7"

# Security
export GEMMA_SECURITY_API_KEY_REQUIRED="false"
export GEMMA_SECURITY_RATE_LIMIT_PER_MINUTE="60"
export GEMMA_SECURITY_ALLOWED_ORIGINS="*"

# Performance
export GEMMA_PERFORMANCE_DEVICE="auto"
export GEMMA_PERFORMANCE_PRECISION="float16"
export GEMMA_CACHE_ENABLED="true"
```

Or create a `.env` file in the project root:

```env
GEMMA_MODEL_NAME=google/gemma-2b-it
GEMMA_SERVER_HOST=localhost
GEMMA_SERVER_PORT=8000
GEMMA_LOG_LEVEL=INFO
```

## 🧪 Testing the Server

### Using the Test Client
```bash
# Run the comprehensive test suite
python examples/server_client.py
```

### Manual Testing with curl

#### Health Check
```bash
curl http://localhost:8000/health
```

#### Chat Completion
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-2b-it",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 50
  }'
```

#### Streaming Chat
```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma-2b-it",
    "messages": [{"role": "user", "content": "Tell me a joke"}],
    "stream": true
  }' \
  --no-buffer
```

## 🔧 Development

### Adding New Endpoints

1. **Define Pydantic models** in `src/server/api/schemas.py`:
```python
class MyRequest(BaseModel):
    query: str = Field(..., description="Query parameter")

class MyResponse(BaseModel):
    result: str = Field(..., description="Result data")
```

2. **Create endpoint** in appropriate router:
```python
@router.post("/my-endpoint", response_model=MyResponse)
async def my_endpoint(request: MyRequest):
    # Implementation
    return MyResponse(result="success")
```

3. **Add router** to main application in `src/server/main.py`:
```python
app.include_router(my_router, prefix="/v1", tags=["my-feature"])
```

### Adding Middleware

Create middleware in `src/server/middleware.py`:
```python
class MyMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Pre-processing
        response = await call_next(request)
        # Post-processing
        return response

def add_my_middleware(app: FastAPI, settings: Settings):
    app.add_middleware(MyMiddleware, settings=settings)
```

### WebSocket Message Types

Add new message types in `src/server/websocket.py`:
```python
async def handle_my_message_type(self, message: WebSocketMessage, connection_id: str):
    # Handle custom message type
    response = WebSocketResponse(
        type="my_response",
        session_id=session_id,
        data={"custom": "data"}
    )
    await manager.send_to_connection(response.json(), connection_id)
```

## 📊 Monitoring & Observability

### Prometheus Metrics

The server exposes Prometheus metrics at `/metrics`:

- `http_requests_total` - Total HTTP requests by method, endpoint, and status
- `http_request_duration_seconds` - Request duration histogram
- `active_connections_total` - Current active connections
- `model_inference_total` - Model inference requests
- `model_inference_duration_seconds` - Model inference duration
- `memory_usage_bytes` - Memory usage by type
- `cpu_usage_percent` - CPU usage percentage

### Health Check Endpoints

- **`/health`** - Comprehensive health check with system status
- **`/health/ready`** - Kubernetes readiness probe
- **`/health/live`** - Kubernetes liveness probe
- **`/health/status`** - Detailed service status
- **`/health/metrics`** - Application performance metrics

### Logging

Structured logging with configurable levels:
```python
# Request logging
INFO - POST /v1/chat/completions - 192.168.1.1 - 200 - 1.234s

# Error logging
ERROR - Failed to generate response: Model not loaded
```

## 🔒 Security Features

### Authentication
- API key-based authentication
- Configurable public endpoints
- Bearer token support

### Rate Limiting
- Sliding window rate limiting
- Per-IP tracking
- Configurable limits and windows
- Automatic blocking for abuse

### CORS
- Configurable allowed origins
- Credential support
- Method and header restrictions

### Input Validation
- Pydantic model validation
- Request size limits
- Content type validation
- XSS protection

## 🐳 Deployment

### Docker
```dockerfile
FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src/ src/
EXPOSE 8000

CMD ["uvicorn", "src.server.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Kubernetes
```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gemma-chatbot
spec:
  replicas: 3
  selector:
    matchLabels:
      app: gemma-chatbot
  template:
    spec:
      containers:
      - name: gemma-chatbot
        image: gemma-chatbot:latest
        ports:
        - containerPort: 8000
        livenessProbe:
          httpGet:
            path: /health/live
            port: 8000
        readinessProbe:
          httpGet:
            path: /health/ready
            port: 8000
        resources:
          requests:
            memory: "2Gi"
            cpu: "1"
          limits:
            memory: "4Gi"
            cpu: "2"
```

### Load Balancer Configuration
- Use `/health/ready` for readiness checks
- Use `/health/live` for liveness checks
- Enable sticky sessions for WebSocket connections
- Configure timeouts appropriately for model inference

## 🔍 Troubleshooting

### Common Issues

1. **Model Loading Errors**
   ```
   Check /health endpoint for model status
   Verify CUDA availability: /health/status
   Check memory usage: /health/metrics
   ```

2. **WebSocket Connection Failures**
   ```
   Verify CORS settings for WebSocket origins
   Check connection limits and timeouts
   Monitor active connections: /health/metrics
   ```

3. **Performance Issues**
   ```
   Monitor metrics: /metrics endpoint
   Check cache hit rates: /health/metrics
   Verify GPU utilization in logs
   ```

4. **Authentication Problems**
   ```
   Verify API key configuration
   Check allowed origins for CORS
   Review authentication logs
   ```

## 📈 Performance Optimization

### Model Optimization
- Use appropriate precision (float16 vs float32)
- Enable optimized attention mechanisms
- Configure batch sizes for throughput
- Implement model warming strategies

### Caching Strategy
- Enable response caching for repeated queries
- Configure appropriate TTL values
- Monitor cache hit rates
- Implement cache eviction policies

### Connection Management
- Configure connection pooling
- Set appropriate timeouts
- Implement connection health checks
- Monitor connection metrics

### Resource Management
- Monitor memory usage patterns
- Configure GPU memory fractions
- Implement resource cleanup
- Use connection limiting

---

## 🤝 Contributing

When contributing to the server implementation:

1. Follow FastAPI best practices
2. Add comprehensive type hints
3. Include proper error handling
4. Write tests for new endpoints
5. Update documentation and examples
6. Ensure backward compatibility

---

This production-ready server provides a solid foundation for deploying Gemma chatbot capabilities with enterprise-grade features, monitoring, and scalability.
