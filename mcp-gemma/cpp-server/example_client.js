/**
 * Example MCP WebSocket Client for testing the Gemma MCP Server
 * Run with: node example_client.js
 */

const WebSocket = require('ws');

class MCPClient {
    constructor(url = 'ws://localhost:8080') {
        this.url = url;
        this.ws = null;
        this.requestId = 0;
        this.pendingRequests = new Map();
    }

    connect() {
        return new Promise((resolve, reject) => {
            console.log(`Connecting to ${this.url}...`);

            this.ws = new WebSocket(this.url, ['mcp']);

            this.ws.on('open', () => {
                console.log('Connected to MCP server');
                this.initialize().then(resolve).catch(reject);
            });

            this.ws.on('message', (data) => {
                this.handleMessage(JSON.parse(data.toString()));
            });

            this.ws.on('close', (code, reason) => {
                console.log(`Connection closed: ${code} ${reason}`);
            });

            this.ws.on('error', (error) => {
                console.error('WebSocket error:', error);
                reject(error);
            });
        });
    }

    handleMessage(message) {
        console.log('Received:', JSON.stringify(message, null, 2));

        if (message.id && this.pendingRequests.has(message.id)) {
            const { resolve, reject } = this.pendingRequests.get(message.id);
            this.pendingRequests.delete(message.id);

            if (message.error) {
                reject(new Error(`${message.error.message}: ${message.error.data || ''}`));
            } else {
                resolve(message.result);
            }
        }

        // Handle server-initiated messages
        if (message.method === 'initialize') {
            // Server sent initialize, respond appropriately
            console.log('Server initiated handshake');
        }
    }

    sendRequest(method, params = {}) {
        return new Promise((resolve, reject) => {
            const id = `req_${++this.requestId}`;
            const request = {
                jsonrpc: '2.0',
                id,
                method,
                params
            };

            this.pendingRequests.set(id, { resolve, reject });

            console.log('Sending:', JSON.stringify(request, null, 2));
            this.ws.send(JSON.stringify(request));

            // Timeout after 30 seconds
            setTimeout(() => {
                if (this.pendingRequests.has(id)) {
                    this.pendingRequests.delete(id);
                    reject(new Error('Request timeout'));
                }
            }, 30000);
        });
    }

    async initialize() {
        console.log('Initializing MCP session...');
        const result = await this.sendRequest('initialize', {
            protocolVersion: '2024-11-05',
            clientInfo: {
                name: 'example-client',
                version: '1.0.0'
            },
            capabilities: {
                tools: {}
            }
        });

        console.log('Initialization successful:', result);

        // Send initialized notification
        const initialized = {
            jsonrpc: '2.0',
            method: 'initialized',
            params: {}
        };
        this.ws.send(JSON.stringify(initialized));

        return result;
    }

    async listTools() {
        console.log('\\nListing available tools...');
        return await this.sendRequest('tools/list');
    }

    async callTool(name, arguments = {}) {
        console.log(`\\nCalling tool: ${name}`);
        return await this.sendRequest('tools/call', {
            name,
            arguments
        });
    }

    async getServerStatus() {
        return await this.callTool('get_server_status');
    }

    async generateText(prompt, options = {}) {
        return await this.callTool('generate_text', {
            prompt,
            ...options
        });
    }

    async getModelInfo() {
        return await this.callTool('get_model_info');
    }

    async tokenizeText(text) {
        return await this.callTool('tokenize_text', { text });
    }

    close() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Example usage
async function main() {
    const client = new MCPClient();

    try {
        // Connect and initialize
        await client.connect();

        // List available tools
        const tools = await client.listTools();
        console.log('\\nAvailable tools:', JSON.stringify(tools, null, 2));

        // Get server status
        const status = await client.getServerStatus();
        console.log('\\nServer status:', JSON.stringify(status, null, 2));

        // Get model info
        try {
            const modelInfo = await client.getModelInfo();
            console.log('\\nModel info:', JSON.stringify(modelInfo, null, 2));
        } catch (error) {
            console.log('\\nModel info error (model may not be loaded):', error.message);
        }

        // Test tokenization
        try {
            const tokens = await client.tokenizeText('Hello, world!');
            console.log('\\nTokenization result:', JSON.stringify(tokens, null, 2));
        } catch (error) {
            console.log('\\nTokenization error:', error.message);
        }

        // Test text generation
        try {
            console.log('\\nGenerating text...');
            const generated = await client.generateText('The future of AI is', {
                max_tokens: 50,
                temperature: 0.7
            });
            console.log('\\nGenerated text:', JSON.stringify(generated, null, 2));
        } catch (error) {
            console.log('\\nText generation error (model may not be loaded):', error.message);
        }

        // Test ping
        try {
            const pingResult = await client.sendRequest('ping');
            console.log('\\nPing result:', JSON.stringify(pingResult, null, 2));
        } catch (error) {
            console.log('\\nPing error:', error.message);
        }

    } catch (error) {
        console.error('Error:', error);
    } finally {
        client.close();
        console.log('\\nClient disconnected');
    }
}

// Handle graceful shutdown
process.on('SIGINT', () => {
    console.log('\\nShutting down...');
    process.exit(0);
});

if (require.main === module) {
    main().catch(console.error);
}

module.exports = MCPClient;