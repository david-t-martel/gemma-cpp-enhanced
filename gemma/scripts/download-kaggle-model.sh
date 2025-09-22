#!/bin/bash
#
# Kaggle Model Downloader Utility
# Downloads LLM models from Kaggle using the kaggle CLI tool or curl fallback
#
# Usage: ./download-kaggle-model.sh [OPTIONS]
# Examples:
#   ./download-kaggle-model.sh --owner google --model gemma --framework tensorRtLlm --variation 2b-it --version 2
#   ./download-kaggle-model.sh --owner google --model gemma-2 --framework gemmaCpp --variation gemma2-2b-it-sfp --version 1
#   ./download-kaggle-model.sh --full-path "google/gemma/tensorRtLlm/2b-it/2"
#   ./download-kaggle-model.sh --full-path "google/gemma-3/gemmaCpp/3.0-4b-it-sfp/1"  # Test model
#
# Curl Fallback:
#   If kaggle CLI is not installed, the script will fall back to using curl.
#   Set FORCE_CURL=1 to force using curl even if kaggle CLI is available.
#   Credentials are sourced from KAGGLE_USERNAME/KAGGLE_KEY env vars or ~/.kaggle/kaggle.json

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Default values
OUTPUT_DIR="${GEMMA_MODEL_DIR:-/c/codedev/llm/.models}"
EXTRACT_ARCHIVE=true
QUIET_MODE=false

# Function to display usage
usage() {
    cat << EOF
Kaggle Model Downloader Utility

USAGE:
    $(basename "$0") [OPTIONS]

OPTIONS:
    -o, --owner OWNER           Model owner/organization (e.g., google)
    -m, --model MODEL           Model name (e.g., gemma, gemma-2)
    -f, --framework FRAMEWORK   Framework type (e.g., tensorRtLlm, gemmaCpp, pytorch)
    -v, --variation VARIATION   Model variation (e.g., 2b-it, 7b-it-sfp)
    -n, --version VERSION       Model version number
    -p, --full-path PATH        Full model path (e.g., "google/gemma/tensorRtLlm/2b-it/2")
    -d, --output-dir DIR        Output directory (default: $OUTPUT_DIR)
    -x, --no-extract           Don't extract downloaded archives
    -q, --quiet                Quiet mode (less verbose output)
    -l, --list                 List available versions for a model
    -h, --help                 Display this help message

EXAMPLES:
    # Download using individual parameters
    $(basename "$0") --owner google --model gemma --framework tensorRtLlm --variation 2b-it --version 2

    # Download using full path
    $(basename "$0") --full-path "google/gemma-2/gemmaCpp/gemma2-2b-it-sfp/1"

    # Download to specific directory
    $(basename "$0") --full-path "google/gemma/pytorch/7b/1" --output-dir ./models

    # List available versions
    $(basename "$0") --owner google --model gemma --list

SUPPORTED MODEL TYPES:
    - Gemma models (google/gemma, google/gemma-2, google/gemma-3)
    - RecurrentGemma (google/recurrentgemma)
    - PaliGemma (google/paligemma, google/paligemma-2)
    - CodeGemma (google/codegemma)
    - Other Kaggle models with proper path structure

ENVIRONMENT VARIABLES:
    KAGGLE_USERNAME     Your Kaggle username (required)
    KAGGLE_KEY          Your Kaggle API key (required)
    GEMMA_MODEL_DIR     Default output directory
    FORCE_CURL          Set to 1 to force using curl instead of kaggle CLI

EOF
}

# Function to load Kaggle credentials
load_kaggle_credentials() {
    # Check for Kaggle credentials in environment variables
    if [[ -z "$KAGGLE_USERNAME" || -z "$KAGGLE_KEY" ]]; then
        # Try to load from ~/.kaggle/kaggle.json
        local kaggle_json="$HOME/.kaggle/kaggle.json"
        
        # Also check Windows path if on WSL/Git Bash
        if [[ ! -f "$kaggle_json" ]] && [[ -f "/mnt/c/Users/$USER/.kaggle/kaggle.json" ]]; then
            kaggle_json="/mnt/c/Users/$USER/.kaggle/kaggle.json"
        fi
        
        if [[ -f "$kaggle_json" ]]; then
            # Try jq first, fall back to grep/sed if not available
            if command -v jq &> /dev/null; then
                export KAGGLE_USERNAME=$(jq -r '.username' "$kaggle_json" 2>/dev/null)
                export KAGGLE_KEY=$(jq -r '.key' "$kaggle_json" 2>/dev/null)
            else
                # Fallback to grep/sed
                export KAGGLE_USERNAME=$(grep -o '"username":"[^"]*' "$kaggle_json" | sed 's/"username":"//')
                export KAGGLE_KEY=$(grep -o '"key":"[^"]*' "$kaggle_json" | sed 's/"key":"//')
            fi
        fi
        
        if [[ -z "$KAGGLE_USERNAME" || -z "$KAGGLE_KEY" ]]; then
            echo -e "${RED}Error: Kaggle credentials not found${NC}"
            echo "Set KAGGLE_USERNAME and KAGGLE_KEY environment variables"
            echo "Or create ~/.kaggle/kaggle.json with your credentials"
            return 1
        fi
    fi
    return 0
}

# Function to check prerequisites
check_prerequisites() {
    # Load credentials first
    if ! load_kaggle_credentials; then
        exit 1
    fi
    
    # Check if we're forcing curl
    if [[ "$FORCE_CURL" == "1" ]]; then
        echo -e "${YELLOW}FORCE_CURL is set, will use curl for download${NC}"
        USE_CURL=true
        return 0
    fi
    
    # Check if kaggle CLI is installed
    if ! command -v kaggle &> /dev/null; then
        echo -e "${YELLOW}kaggle CLI not found, will use curl fallback${NC}"
        USE_CURL=true
        return 0
    fi
    
    USE_CURL=false
}

# Function to list model versions
list_versions() {
    local owner="$1"
    local model="$2"

    echo -e "${GREEN}Fetching versions for $owner/$model...${NC}"
    kaggle models instances versions list "$owner/$model" 2>/dev/null || {
        echo -e "${RED}Error: Failed to list versions for $owner/$model${NC}"
        echo "Make sure the model path is correct"
        exit 1
    }
}

# Function to download model using curl
download_model_with_curl() {
    local model_path="$1"
    local output_dir="$2"
    
    # Parse the model path for API v1 format
    # Format: owner/model/framework/variation/version
    IFS='/' read -ra PARTS <<< "$model_path"
    local owner="${PARTS[0]}"
    local model="${PARTS[1]}"
    local framework="${PARTS[2]:-}"
    local variation="${PARTS[3]:-}"
    local version="${PARTS[4]:-1}"
    
    # Construct the download directory name
    local download_name="${model}"
    [[ -n "$framework" ]] && download_name="${download_name}-${framework}"
    [[ -n "$variation" ]] && download_name="${download_name}-${variation}"
    [[ -n "$version" ]] && download_name="${download_name}-v${version}"
    
    local target_dir="$output_dir/$download_name"
    
    # Create output directory
    mkdir -p "$target_dir"
    
    echo -e "${GREEN}Downloading model using curl: $model_path${NC}"
    echo -e "${YELLOW}Target directory: $target_dir${NC}"
    
    # Construct the API URL
    # API v1 format: https://www.kaggle.com/api/v1/models/{owner}/{model}/{framework}/{variation}/{version}/download
    local api_url="https://www.kaggle.com/api/v1/models/${owner}/${model}/${framework}/${variation}/${version}/download"
    
    echo -e "${YELLOW}Downloading from: $api_url${NC}"
    
    # Download the model archive
    local archive_file="$target_dir/model.tar.gz"
    
    # Use curl with authentication
    curl -L \
        -u "${KAGGLE_USERNAME}:${KAGGLE_KEY}" \
        -o "$archive_file" \
        --fail \
        --retry 3 \
        --retry-delay 5 \
        --continue-at - \
        -H "User-Agent: kaggle-model-downloader/1.0" \
        "$api_url" || {
        echo -e "${RED}Error: Failed to download model using curl${NC}"
        echo "Please check:"
        echo "  - Your credentials are correct"
        echo "  - The model path is valid: $model_path"
        echo "  - You have access to this model"
        return 1
    }
    
    # Extract the archive if requested
    if [[ "$EXTRACT_ARCHIVE" == true ]] && [[ -f "$archive_file" ]]; then
        echo -e "${YELLOW}Extracting archive...${NC}"
        cd "$target_dir"
        
        # Extract based on file type
        if [[ "$archive_file" == *.tar.gz ]]; then
            tar -xzf "$archive_file"
            echo "Extracted $archive_file successfully"
        elif [[ "$archive_file" == *.zip ]]; then
            unzip -q "$archive_file"
            echo "Extracted $archive_file successfully"
        fi
        
        cd - > /dev/null
    fi
    
    echo -e "${GREEN}✓ Model downloaded successfully to: $target_dir${NC}"
    
    # List the downloaded files
    echo -e "\n${YELLOW}Downloaded files:${NC}"
    ls -la "$target_dir" | head -20
    
    return 0
}

# Function to download model using kaggle CLI
download_model_with_kaggle_cli() {
    local model_path="$1"
    local output_dir="$2"
    
    # Parse the model path
    IFS='/' read -ra PARTS <<< "$model_path"
    local owner="${PARTS[0]}"
    local model="${PARTS[1]}"
    local framework="${PARTS[2]:-}"
    local variation="${PARTS[3]:-}"
    local version="${PARTS[4]:-}"
    
    # Construct the download directory name
    local download_name="${model}"
    [[ -n "$framework" ]] && download_name="${download_name}-${framework}"
    [[ -n "$variation" ]] && download_name="${download_name}-${variation}"
    [[ -n "$version" ]] && download_name="${download_name}-v${version}"
    
    local target_dir="$output_dir/$download_name"
    
    echo -e "${GREEN}Downloading model using kaggle CLI: $model_path${NC}"
    echo -e "${YELLOW}Target directory: $target_dir${NC}"
    
    # Download the model
    if [[ "$QUIET_MODE" == true ]]; then
        kaggle models instances versions download "$model_path" \
            --path "$target_dir" \
            --unzip 2>/dev/null || {
            echo -e "${RED}Error: Download failed with kaggle CLI${NC}"
            return 1
        }
    else
        kaggle models instances versions download "$model_path" \
            --path "$target_dir" \
            --unzip || {
            echo -e "${RED}Error: Download failed with kaggle CLI${NC}"
            return 1
        }
    fi
    
    # Extract archives if requested and kaggle didn't auto-extract
    if [[ "$EXTRACT_ARCHIVE" == true ]]; then
        echo -e "${YELLOW}Checking for archives to extract...${NC}"
        cd "$target_dir"
        
        # Extract .tar.gz files
        for archive in *.tar.gz; do
            if [[ -f "$archive" ]]; then
                echo "Extracting $archive..."
                tar -xzf "$archive"
                echo "Extracted $archive successfully"
            fi
        done
        
        # Extract .zip files
        for archive in *.zip; do
            if [[ -f "$archive" ]]; then
                echo "Extracting $archive..."
                unzip -q "$archive"
                echo "Extracted $archive successfully"
            fi
        done
        
        cd - > /dev/null
    fi
    
    echo -e "${GREEN}✓ Model downloaded successfully to: $target_dir${NC}"
    
    # List the downloaded files
    echo -e "\n${YELLOW}Downloaded files:${NC}"
    ls -la "$target_dir" | head -20
    
    return 0
}

# Function to download model
download_model() {
    local model_path="$1"
    local output_dir="$2"

    # Decide which download method to use
    if [[ "$USE_CURL" == true ]]; then
        download_model_with_curl "$model_path" "$output_dir"
    else
        # Try kaggle CLI first
        if ! download_model_with_kaggle_cli "$model_path" "$output_dir"; then
            echo -e "${YELLOW}Kaggle CLI failed, trying curl fallback...${NC}"
            if ! download_model_with_curl "$model_path" "$output_dir"; then
                echo -e "${RED}Error: Both download methods failed${NC}"
                exit 1
            fi
        fi
    fi
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -o|--owner)
            OWNER="$2"
            shift 2
            ;;
        -m|--model)
            MODEL="$2"
            shift 2
            ;;
        -f|--framework)
            FRAMEWORK="$2"
            shift 2
            ;;
        -v|--variation)
            VARIATION="$2"
            shift 2
            ;;
        -n|--version)
            VERSION="$2"
            shift 2
            ;;
        -p|--full-path)
            FULL_PATH="$2"
            shift 2
            ;;
        -d|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -x|--no-extract)
            EXTRACT_ARCHIVE=false
            shift
            ;;
        -q|--quiet)
            QUIET_MODE=true
            shift
            ;;
        -l|--list)
            LIST_MODE=true
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo -e "${RED}Unknown option: $1${NC}"
            usage
            exit 1
            ;;
    esac
done

# Check prerequisites
check_prerequisites

# Handle list mode
if [[ "$LIST_MODE" == true ]]; then
    if [[ -z "$OWNER" || -z "$MODEL" ]]; then
        echo -e "${RED}Error: --owner and --model required for listing versions${NC}"
        exit 1
    fi
    list_versions "$OWNER" "$MODEL"
    exit 0
fi

# Construct the model path
if [[ -n "$FULL_PATH" ]]; then
    MODEL_PATH="$FULL_PATH"
elif [[ -n "$OWNER" && -n "$MODEL" ]]; then
    MODEL_PATH="$OWNER/$MODEL"
    [[ -n "$FRAMEWORK" ]] && MODEL_PATH="$MODEL_PATH/$FRAMEWORK"
    [[ -n "$VARIATION" ]] && MODEL_PATH="$MODEL_PATH/$VARIATION"
    [[ -n "$VERSION" ]] && MODEL_PATH="$MODEL_PATH/$VERSION"
else
    echo -e "${RED}Error: Either --full-path or --owner/--model parameters required${NC}"
    usage
    exit 1
fi

# Download the model
download_model "$MODEL_PATH" "$OUTPUT_DIR"

# Create a convenience symlink for gemma.cpp if it's a compatible model
if [[ "$MODEL_PATH" == *"gemmaCpp"* ]] || [[ "$MODEL_PATH" == *"gemma"*"cpp"* ]]; then
    echo -e "\n${YELLOW}Tip: For gemma.cpp, you can use these files with:${NC}"
    echo "  ./gemma --weights [path-to-.sbs-file] --tokenizer [path-to-tokenizer.spm]"
fi