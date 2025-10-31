"""Data fixtures for testing.

This module provides static data fixtures including sample configurations,
prompts, responses, and other test data. All data is realistic and representative
of actual usage patterns.
"""

from typing import Any, Dict, List, Tuple

# =============================================================================
# Sample Configurations
# =============================================================================

SAMPLE_CONFIGS = {
    "minimal": {
        "gemma": {
            "default_model": "C:/codedev/llm/.models/gemma-2b-it.sbs",
            "default_tokenizer": "C:/codedev/llm/.models/tokenizer.spm",
            "executable": "C:/codedev/llm/gemma/build/Release/gemma.exe",
        },
    },
    "development": {
        "gemma": {
            "default_model": "C:/codedev/llm/.models/gemma-2b-it.sbs",
            "default_tokenizer": "C:/codedev/llm/.models/tokenizer.spm",
            "executable": "C:/codedev/llm/gemma/build/Release/gemma.exe",
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "pool_size": 10,
            "enable_fallback": True,
        },
        "memory": {
            "working_ttl": 900,
            "short_term_ttl": 3600,
            "long_term_ttl": 2592000,
            "working_capacity": 15,
            "short_term_capacity": 100,
        },
        "ui": {
            "theme": "default",
            "show_timestamps": True,
            "syntax_highlighting": True,
        },
    },
    "production": {
        "gemma": {
            "default_model": "C:/codedev/llm/.models/gemma-4b-it-sfp.sbs",
            "default_tokenizer": "C:/codedev/llm/.models/tokenizer.spm",
            "executable": "C:/codedev/llm/gemma/build/Release/gemma.exe",
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 0,
            "pool_size": 20,
            "enable_fallback": True,
            "connection_timeout": 5,
            "command_timeout": 10,
        },
        "memory": {
            "working_ttl": 900,
            "short_term_ttl": 3600,
            "long_term_ttl": 2592000,
            "episodic_ttl": 604800,
            "semantic_ttl": 0,
            "working_capacity": 20,
            "short_term_capacity": 200,
            "long_term_capacity": 10000,
            "enable_background_tasks": True,
            "auto_consolidate": True,
        },
        "embedding": {
            "provider": "local",
            "model": "all-MiniLM-L6-v2",
            "dimension": 384,
            "batch_size": 64,
            "cache_embeddings": True,
        },
        "vector_store": {
            "dimension": 384,
            "index_type": "flat",
            "metric": "cosine",
            "enable_quantization": False,
        },
        "ui": {
            "theme": "dark",
            "show_timestamps": True,
            "syntax_highlighting": True,
            "markdown_rendering": True,
        },
    },
    "testing": {
        "gemma": {
            "default_model": "C:/test/models/mock-model.sbs",
            "default_tokenizer": "C:/test/models/mock-tokenizer.spm",
            "executable": "C:/test/mock-gemma.exe",
        },
        "redis": {
            "host": "localhost",
            "port": 6379,
            "db": 1,  # Use separate DB for testing
            "pool_size": 5,
            "enable_fallback": True,
        },
        "memory": {
            "working_ttl": 60,  # Shorter TTLs for testing
            "short_term_ttl": 300,
            "long_term_ttl": 3600,
            "working_capacity": 5,
            "short_term_capacity": 20,
        },
        "ui": {
            "theme": "minimal",
            "show_timestamps": False,
            "syntax_highlighting": False,
        },
    },
}

# =============================================================================
# Sample Prompts
# =============================================================================

SAMPLE_PROMPTS = {
    "simple": [
        "What is 2 + 2?",
        "Tell me a joke.",
        "Hello!",
        "How are you?",
        "What's your name?",
    ],
    "factual": [
        "What is the capital of France?",
        "Who wrote Romeo and Juliet?",
        "When did World War 2 end?",
        "What is the speed of light?",
        "How many planets are in the solar system?",
    ],
    "technical": [
        "Explain how a neural network works.",
        "What is the difference between Python and JavaScript?",
        "How does blockchain technology work?",
        "Explain quantum computing in simple terms.",
        "What are the SOLID principles in software development?",
    ],
    "coding": [
        "Write a Python function to calculate factorial.",
        "How do I sort a list in Python?",
        "Explain what a closure is in JavaScript.",
        "Write a SQL query to find duplicate records.",
        "How do I implement a binary search tree?",
    ],
    "creative": [
        "Write a short poem about technology.",
        "Describe a sunset in vivid detail.",
        "Create a short story about a time traveler.",
        "Write a haiku about programming.",
        "Compose a limerick about artificial intelligence.",
    ],
    "conversational": [
        "Can you help me with my homework?",
        "I'm feeling stressed. What should I do?",
        "Tell me something interesting about space.",
        "What's the best way to learn programming?",
        "I need advice on starting a business.",
    ],
}

# =============================================================================
# Sample Responses
# =============================================================================

SAMPLE_RESPONSES = {
    "simple": [
        "2 + 2 equals 4.",
        "Why don't scientists trust atoms? Because they make up everything!",
        "Hello! How can I help you today?",
        "I'm doing well, thank you for asking!",
        "I'm an AI assistant created to help answer questions and have conversations.",
    ],
    "factual": [
        "The capital of France is Paris, located in the north-central part of the country.",
        "Romeo and Juliet was written by William Shakespeare around 1594-1595.",
        "World War 2 ended on September 2, 1945, with Japan's formal surrender.",
        "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
        "There are 8 planets in our solar system: Mercury, Venus, Earth, Mars, Jupiter, Saturn, Uranus, and Neptune.",
    ],
    "technical": [
        """A neural network is a computational model inspired by biological neural networks. It consists of:
1. Input layer: Receives data
2. Hidden layers: Process information through weighted connections
3. Output layer: Produces predictions

The network learns by adjusting weights through backpropagation to minimize error.""",
        """Python and JavaScript differ in several ways:
- Python: General-purpose, server-side, synchronous by default
- JavaScript: Web-focused, runs in browsers and Node.js, asynchronous
- Python: Indentation-based syntax
- JavaScript: Bracket-based syntax
Both are dynamically typed and widely used.""",
        """Blockchain is a distributed ledger technology where:
1. Transactions are grouped into blocks
2. Each block contains a cryptographic hash of the previous block
3. This creates an immutable chain
4. Multiple nodes maintain copies, ensuring decentralization
5. Consensus mechanisms validate new blocks""",
        """Quantum computing uses quantum mechanical phenomena like:
- Superposition: Qubits exist in multiple states simultaneously
- Entanglement: Qubits are correlated regardless of distance
This allows quantum computers to solve certain problems exponentially faster than classical computers.""",
        """The SOLID principles are:
1. Single Responsibility: One class, one purpose
2. Open/Closed: Open for extension, closed for modification
3. Liskov Substitution: Subtypes must be substitutable for base types
4. Interface Segregation: Many specific interfaces > one general
5. Dependency Inversion: Depend on abstractions, not concretions""",
    ],
    "coding": [
        """Here's a factorial function in Python:

def factorial(n):
    if n <= 1:
        return 1
    return n * factorial(n - 1)

# Or iteratively:
def factorial_iter(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result""",
        """There are several ways to sort a list in Python:

# In-place sorting
my_list.sort()

# Create new sorted list
sorted_list = sorted(my_list)

# Reverse order
my_list.sort(reverse=True)

# Custom key
my_list.sort(key=lambda x: x.lower())""",
        """A closure in JavaScript is a function that retains access to variables from its outer scope:

function makeCounter() {
    let count = 0;
    return function() {
        count++;
        return count;
    };
}

const counter = makeCounter();
console.log(counter()); // 1
console.log(counter()); // 2""",
        """SQL query to find duplicate records:

SELECT column_name, COUNT(*)
FROM table_name
GROUP BY column_name
HAVING COUNT(*) > 1;

-- For multiple columns:
SELECT col1, col2, COUNT(*)
FROM table_name
GROUP BY col1, col2
HAVING COUNT(*) > 1;""",
        """Binary Search Tree implementation:

class Node:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, value):
        if not self.root:
            self.root = Node(value)
        else:
            self._insert_recursive(self.root, value)

    def _insert_recursive(self, node, value):
        if value < node.value:
            if node.left is None:
                node.left = Node(value)
            else:
                self._insert_recursive(node.left, value)
        else:
            if node.right is None:
                node.right = Node(value)
            else:
                self._insert_recursive(node.right, value)""",
    ],
}

# =============================================================================
# Sample Conversation Histories
# =============================================================================

SAMPLE_CONVERSATIONS = [
    {
        "title": "Python Help",
        "messages": [
            {"role": "user", "content": "I'm learning Python. Where should I start?"},
            {
                "role": "assistant",
                "content": "Great choice! Start with basics: variables, data types, loops, and functions. Use resources like Python.org tutorials.",
            },
            {"role": "user", "content": "What's a good first project?"},
            {
                "role": "assistant",
                "content": "Try building a calculator, todo list, or simple text-based game. These cover fundamental concepts.",
            },
        ],
    },
    {
        "title": "Technical Discussion",
        "messages": [
            {"role": "user", "content": "Explain the difference between async and sync code."},
            {
                "role": "assistant",
                "content": "Synchronous code executes line by line, blocking until each operation completes. Asynchronous code allows other operations to run while waiting for I/O.",
            },
            {"role": "user", "content": "When should I use async?"},
            {
                "role": "assistant",
                "content": "Use async for I/O-bound operations like network requests, file operations, or database queries. It improves responsiveness.",
            },
        ],
    },
    {
        "title": "Debugging Help",
        "messages": [
            {"role": "user", "content": "My code throws a KeyError. What does that mean?"},
            {
                "role": "assistant",
                "content": "KeyError means you tried to access a dictionary key that doesn't exist. Use .get() method or check if key exists first.",
            },
            {"role": "user", "content": "How do I check if a key exists?"},
            {
                "role": "assistant",
                "content": "Use: if 'key' in my_dict: or value = my_dict.get('key', default_value)",
            },
        ],
    },
]

# =============================================================================
# Sample Model Presets
# =============================================================================

SAMPLE_MODEL_PRESETS = {
    "gemma-2b-fast": {
        "name": "gemma-2b-fast",
        "model_path": "C:/codedev/llm/.models/gemma-2b-it.sbs",
        "tokenizer_path": "C:/codedev/llm/.models/tokenizer.spm",
        "max_tokens": 2048,
        "temperature": 0.7,
        "top_p": 0.9,
        "top_k": 40,
        "description": "Fast 2B model for quick responses",
        "recommended_use_cases": ["development", "testing", "quick queries"],
    },
    "gemma-4b-balanced": {
        "name": "gemma-4b-balanced",
        "model_path": "C:/codedev/llm/.models/gemma-4b-it-sfp.sbs",
        "tokenizer_path": "C:/codedev/llm/.models/tokenizer.spm",
        "max_tokens": 4096,
        "temperature": 0.8,
        "top_p": 0.95,
        "top_k": 50,
        "description": "Balanced 4B model for quality responses",
        "recommended_use_cases": ["general", "coding", "explanations"],
    },
    "gemma-9b-quality": {
        "name": "gemma-9b-quality",
        "model_path": "C:/codedev/llm/.models/gemma-9b-it-sfp.sbs",
        "tokenizer_path": "C:/codedev/llm/.models/tokenizer.spm",
        "max_tokens": 8192,
        "temperature": 0.9,
        "top_p": 0.95,
        "top_k": 60,
        "description": "High-quality 9B model for complex tasks",
        "recommended_use_cases": ["research", "analysis", "creative writing"],
    },
}

# =============================================================================
# Sample Performance Profiles
# =============================================================================

SAMPLE_PERFORMANCE_PROFILES = {
    "fast": {
        "name": "fast",
        "description": "Optimized for speed",
        "batch_size": 16,
        "cache_size": 500,
        "max_concurrent_requests": 5,
        "timeout_seconds": 15.0,
        "enable_caching": True,
        "enable_batching": False,
    },
    "balanced": {
        "name": "balanced",
        "description": "Balanced performance and quality",
        "batch_size": 32,
        "cache_size": 1000,
        "max_concurrent_requests": 10,
        "timeout_seconds": 30.0,
        "enable_caching": True,
        "enable_batching": True,
    },
    "quality": {
        "name": "quality",
        "description": "Optimized for quality",
        "batch_size": 64,
        "cache_size": 2000,
        "max_concurrent_requests": 20,
        "timeout_seconds": 60.0,
        "enable_caching": True,
        "enable_batching": True,
    },
}

# =============================================================================
# Sample Error Cases
# =============================================================================

SAMPLE_ERROR_CASES = {
    "invalid_prompts": [
        "",  # Empty prompt
        " " * 100,  # Whitespace only
        "x" * 100000,  # Too long
        "test\x00null",  # Null byte
        "test\x1bescape",  # Escape sequence
    ],
    "invalid_model_paths": [
        "",  # Empty path
        "nonexistent.sbs",  # Doesn't exist
        "model.txt",  # Wrong extension
        "/invalid/path/model.sbs",  # Invalid path
    ],
    "invalid_configs": [
        {},  # Empty config
        {"unknown_section": {}},  # Invalid section
        {"gemma": {"invalid_key": "value"}},  # Invalid key
        {"redis": {"port": "invalid"}},  # Wrong type
    ],
}

# =============================================================================
# Sample Test Data
# =============================================================================

SAMPLE_TEST_DATA = {
    "embeddings": {
        "dimension": 384,
        "sample_vectors": [
            [0.1] * 384,  # Simple uniform vector
            [i / 384 for i in range(384)],  # Linear progression
            [(-1) ** i * 0.5 for i in range(384)],  # Alternating
        ],
    },
    "redis_keys": {
        "memory": [
            "memory:working:msg_001",
            "memory:short_term:msg_002",
            "memory:long_term:msg_003",
        ],
        "embeddings": [
            "embedding:prompt_001",
            "embedding:response_001",
        ],
        "cache": [
            "cache:prompt:hash_001",
            "cache:response:hash_001",
        ],
    },
    "file_paths": {
        "models": [
            "C:/codedev/llm/.models/gemma-2b-it.sbs",
            "C:/codedev/llm/.models/gemma-4b-it-sfp.sbs",
            "C:/codedev/llm/.models/gemma-9b-it-sfp.sbs",
        ],
        "tokenizers": [
            "C:/codedev/llm/.models/tokenizer.spm",
        ],
        "configs": [
            "C:/codedev/llm/gemma/config/config.toml",
            "C:/codedev/llm/gemma/config/mcp_servers.toml",
        ],
    },
}

# =============================================================================
# Helper Functions
# =============================================================================


def get_sample_config(config_type: str = "development") -> Dict[str, Any]:
    """Get a sample configuration by type."""
    return SAMPLE_CONFIGS.get(config_type, SAMPLE_CONFIGS["minimal"])


def get_sample_prompts(category: str = "simple") -> List[str]:
    """Get sample prompts by category."""
    return SAMPLE_PROMPTS.get(category, SAMPLE_PROMPTS["simple"])


def get_sample_responses(category: str = "simple") -> List[str]:
    """Get sample responses by category."""
    return SAMPLE_RESPONSES.get(category, SAMPLE_RESPONSES["simple"])


def get_sample_conversation(index: int = 0) -> Dict[str, Any]:
    """Get a sample conversation by index."""
    if 0 <= index < len(SAMPLE_CONVERSATIONS):
        return SAMPLE_CONVERSATIONS[index]
    return SAMPLE_CONVERSATIONS[0]


def get_sample_model_preset(name: str = "gemma-2b-fast") -> Dict[str, Any]:
    """Get a sample model preset by name."""
    return SAMPLE_MODEL_PRESETS.get(name, SAMPLE_MODEL_PRESETS["gemma-2b-fast"])


def get_sample_performance_profile(name: str = "balanced") -> Dict[str, Any]:
    """Get a sample performance profile by name."""
    return SAMPLE_PERFORMANCE_PROFILES.get(name, SAMPLE_PERFORMANCE_PROFILES["balanced"])
