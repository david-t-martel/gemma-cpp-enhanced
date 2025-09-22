"""
Gemma Rust Extensions

High-performance Rust extensions for Gemma chatbot operations.
"""

try:
    from ._gemma_extensions import *
except ImportError as e:
    raise ImportError(
        f"Failed to import Rust extensions. Make sure they are compiled correctly. Error: {e}"
    )

__version__ = get_version()
__all__ = [
    # Core functions
    "get_version",
    "get_build_info",
    "check_simd_support",
    "warmup",
    "benchmark_operations",
    # Tokenizer
    "FastTokenizer",
    "TokenizerConfig",
    "TokenizationResult",
    "create_bpe_tokenizer",
    "batch_encode",
    "batch_decode",
    # Document Processor
    "DocumentProcessor",
    "DocumentConfig",
    "DocumentFormat",
    "DocumentMetadata",
    "ProcessingResult",
    "process_document",
    "process_documents_batch",
    "detect_format",
    # Tensor operations (commented out)
    # 'TensorOperations',
    # 'simd_dot_product',
    # 'simd_vector_add',
    # 'simd_softmax',
    # 'optimize_attention_weights',
    # 'batch_matmul',
    # 'fast_layer_norm',
    # Cache (commented out)
    # 'LRUCache',
    # 'CacheManager',
    # 'CacheStats',
]
