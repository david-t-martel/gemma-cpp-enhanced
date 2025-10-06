#!/bin/bash

# Simple test script for gemma binary
echo "Testing gemma.cpp binary..."

BINARY="/mnt/c/codedev/llm/gemma/gemma.cpp/build_wsl/gemma"
MODEL="/mnt/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/2b-it.sbs"
TOKENIZER="/mnt/c/codedev/llm/.models/gemma-gemmacpp-2b-it-v3/tokenizer.spm"

if [ ! -f "$BINARY" ]; then
    echo "Error: Binary not found at $BINARY"
    exit 1
fi

if [ ! -f "$MODEL" ]; then
    echo "Error: Model not found at $MODEL"
    exit 1
fi

echo "Binary and model found. Running test..."
echo "What is 2+2?" | $BINARY --model $MODEL --tokenizer $TOKENIZER --max_tokens 20 2>&1 | grep -E "(tokens/sec|Generated)"
