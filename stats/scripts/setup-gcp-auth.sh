#!/bin/bash

# GCP Authentication Setup Script
# This script configures Google Cloud Platform authentication for the gcp-profile business service account

set -euo pipefail

# Colors for output
readonly RED='\033[0;31m'
readonly GREEN='\033[0;32m'
readonly YELLOW='\033[1;33m'
readonly BLUE='\033[0;34m'
readonly CYAN='\033[0;36m'
readonly NC='\033[0m' # No Color

# Script configuration
readonly SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
readonly CONFIG_DIR="$PROJECT_ROOT/config"
readonly ENV_FILE="$PROJECT_ROOT/.env"

# Expected service account configurations
readonly SA_NAME="gcp-profile"
readonly SA_FILE_PATTERNS=("gcp-profile*.json" "gcp-profile*credentials*.json" "*gcp-profile*.json")

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$@${NC}"
}

# Function to print section header
print_header() {
    echo
    print_color $BLUE "================================================"
    print_color $BLUE "    $1"
    print_color $BLUE "================================================"
    echo
}

# Function to check if command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Function to check if running in WSL
is_wsl() {
    [[ -f /proc/version ]] && grep -q "Microsoft\|WSL" /proc/version
}

# Function to find gcp-profile service account files
find_gcp_profile_sa() {
    local locations=(
        "$HOME/.config/gcloud"
        "$HOME/.gcp"
        "$HOME/Downloads"
        "$PROJECT_ROOT"
        "$CONFIG_DIR"
        "/etc/gcp"
        "$HOME/Documents"
    )

    # Add Windows-specific locations if running on WSL
    if is_wsl; then
        local win_home="/mnt/c/Users/$USER"
        locations+=(
            "$win_home/Downloads"
            "$win_home/Documents"
            "$win_home/.gcp"
        )
    fi

    local found_files=()

    for location in "${locations[@]}"; do
        if [[ -d "$location" ]]; then
            for pattern in "${SA_FILE_PATTERNS[@]}"; do
                while IFS= read -r -d '' file; do
                    if [[ -f "$file" ]]; then
                        found_files+=("$file")
                    fi
                done < <(find "$location" -maxdepth 2 -name "$pattern" -type f -print0 2>/dev/null || true)
            done
        fi
    done

    printf '%s\n' "${found_files[@]}"
}

# Function to validate service account file
validate_sa_file() {
    local file="$1"

    if [[ ! -f "$file" ]]; then
        return 1
    fi

    # Check if it's valid JSON
    if ! python3 -c "import json; json.load(open('$file'))" 2>/dev/null; then
        return 1
    fi

    # Check if it has required service account fields
    python3 -c "
import json
import sys

try:
    with open('$file') as f:
        data = json.load(f)

    required_fields = ['type', 'project_id', 'private_key_id', 'private_key', 'client_email', 'client_id']
    missing_fields = [field for field in required_fields if field not in data]

    if missing_fields:
        print(f'Missing required fields: {missing_fields}', file=sys.stderr)
        sys.exit(1)

    if data.get('type') != 'service_account':
        print(f'Not a service account key (type: {data.get(\"type\")})', file=sys.stderr)
        sys.exit(1)

    # Check if it's the gcp-profile account
    client_email = data.get('client_email', '')
    if 'gcp-profile' in client_email or 'business' in client_email:
        print('✓ Found gcp-profile business service account')
    else:
        print(f'⚠ Service account may not be gcp-profile (email: {client_email})')

    print(f'Project ID: {data.get(\"project_id\")}')
    print(f'Client Email: {client_email}')

except Exception as e:
    print(f'Error validating service account file: {e}', file=sys.stderr)
    sys.exit(1)
" 2>/dev/null
}

# Function to check gcloud authentication
check_gcloud_auth() {
    if ! command_exists gcloud; then
        return 1
    fi

    local current_account
    current_account=$(gcloud auth list --filter="status:ACTIVE" --format="value(account)" 2>/dev/null | head -n1)

    if [[ -n "$current_account" ]]; then
        print_color $GREEN "✓ gcloud authenticated as: $current_account"

        local project_id
        project_id=$(gcloud config get-value project 2>/dev/null || echo "")
        if [[ -n "$project_id" ]]; then
            print_color $GREEN "✓ Default project: $project_id"
        fi
        return 0
    fi

    return 1
}

# Function to check Application Default Credentials
check_adc() {
    local adc_file="$HOME/.config/gcloud/application_default_credentials.json"

    if [[ -f "$adc_file" ]]; then
        print_color $GREEN "✓ Application Default Credentials found"

        # Try to extract project info if available
        local project_id
        project_id=$(python3 -c "
import json
try:
    with open('$adc_file') as f:
        data = json.load(f)
    project_id = data.get('quota_project_id') or data.get('project_id')
    if project_id:
        print(project_id)
except:
    pass
" 2>/dev/null || echo "")

        if [[ -n "$project_id" ]]; then
            print_color $GREEN "  Project ID: $project_id"
        fi
        return 0
    fi

    return 1
}

# Function to setup service account authentication
setup_service_account_auth() {
    local sa_file="$1"
    local copy_to_config="${2:-true}"

    # Copy to config directory if requested
    if [[ "$copy_to_config" == "true" ]]; then
        mkdir -p "$CONFIG_DIR"
        local config_sa_file="$CONFIG_DIR/gcp-profile-sa.json"

        if cp "$sa_file" "$config_sa_file"; then
            chmod 600 "$config_sa_file"
            print_color $GREEN "✓ Service account key copied to: $config_sa_file"
            sa_file="$config_sa_file"
        else
            print_color $YELLOW "⚠ Could not copy to config directory, using original location"
        fi
    fi

    # Add to environment file
    if [[ -f "$ENV_FILE" ]]; then
        # Remove existing GOOGLE_APPLICATION_CREDENTIALS if present
        sed -i.bak '/^GOOGLE_APPLICATION_CREDENTIALS=/d' "$ENV_FILE" 2>/dev/null || true
    fi

    echo "GOOGLE_APPLICATION_CREDENTIALS=\"$sa_file\"" >> "$ENV_FILE"
    print_color $GREEN "✓ Added GOOGLE_APPLICATION_CREDENTIALS to .env file"

    # Set for current session
    export GOOGLE_APPLICATION_CREDENTIALS="$sa_file"

    # Extract and set other variables
    python3 -c "
import json
import os

try:
    with open('$sa_file') as f:
        data = json.load(f)

    env_vars = {
        'GCP_PROJECT_ID': data['project_id'],
        'GCP_CLIENT_EMAIL': data['client_email'],
        'GCP_AUTH_METHOD': 'service_account'
    }

    # Add to .env file
    with open('$ENV_FILE', 'a') as f:
        for key, value in env_vars.items():
            # Check if already exists
            existing = False
            if os.path.exists('$ENV_FILE'):
                with open('$ENV_FILE', 'r') as rf:
                    for line in rf:
                        if line.startswith(f'{key}='):
                            existing = True
                            break

            if not existing:
                f.write(f'{key}=\"{value}\"\n')

    print(f'✓ Project ID: {data[\"project_id\"]}')
    print(f'✓ Service Account: {data[\"client_email\"]}')

except Exception as e:
    print(f'Error processing service account: {e}')
    exit(1)
"
}

# Function to setup ADC authentication
setup_adc_auth() {
    if ! command_exists gcloud; then
        print_color $RED "✗ gcloud CLI is required for Application Default Credentials"
        print_color $YELLOW "Install from: https://cloud.google.com/sdk/docs/install"
        return 1
    fi

    print_color $BLUE "Setting up Application Default Credentials..."
    print_color $YELLOW "This will open a browser window for authentication."

    if gcloud auth application-default login; then
        print_color $GREEN "✓ Application Default Credentials configured"

        # Add to environment
        echo "GCP_AUTH_METHOD=\"application_default\"" >> "$ENV_FILE"

        return 0
    else
        print_color $RED "✗ Failed to configure Application Default Credentials"
        return 1
    fi
}

# Function to test authentication
test_authentication() {
    print_color $BLUE "Testing GCP authentication..."

    # Source environment if available
    if [[ -f "$ENV_FILE" ]]; then
        set -a  # Automatically export all variables
        source "$ENV_FILE" 2>/dev/null || true
        set +a
    fi

    python3 -c "
import sys
import os
sys.path.insert(0, '$PROJECT_ROOT/src')

try:
    from gcp import GCPConfig
    from gcp.auth import GCPAuthManager

    # Try to load configuration
    config = None
    try:
        config = GCPConfig.from_env()
        print(f'✓ Configuration loaded from environment')
        print(f'  Project ID: {config.project_id}')
        print(f'  Region: {config.region.value}')
        print(f'  Auth Method: {config.auth_method.value}')
    except Exception as e:
        print(f'⚠ Could not load full config from environment: {e}')
        # Try basic project detection
        project_id = os.getenv('GCP_PROJECT_ID') or os.getenv('GOOGLE_CLOUD_PROJECT')
        if project_id:
            print(f'  Found project ID: {project_id}')

    # Test authentication
    if config:
        auth_manager = GCPAuthManager(config)

        try:
            if auth_manager.validate_credentials():
                print('✓ Authentication successful')
                project_id = auth_manager.get_project_id()
                if project_id:
                    print(f'  Authenticated project: {project_id}')
            else:
                print('✗ Authentication validation failed')
                sys.exit(1)
        except Exception as e:
            print(f'✗ Authentication error: {e}')
            sys.exit(1)
    else:
        print('⚠ Could not create full authentication manager')

except ImportError as e:
    print(f'⚠ Could not import GCP modules: {e}')
    print('  Make sure to install dependencies: uv pip install -r requirements.txt')
except Exception as e:
    print(f'✗ Error testing authentication: {e}')
    sys.exit(1)
"

    if [[ $? -eq 0 ]]; then
        print_color $GREEN "✓ Authentication test passed"
        return 0
    else
        print_color $RED "✗ Authentication test failed"
        return 1
    fi
}

# Main execution
main() {
    print_header "GCP Authentication Setup"

    print_color $CYAN "This script will configure GCP authentication for the gcp-profile business service account."
    print_color $CYAN "It will search for existing credentials and help you set them up."
    echo

    # Create necessary directories
    mkdir -p "$CONFIG_DIR"

    # Check current authentication status
    print_color $BLUE "Checking current authentication status..."

    local has_gcloud_auth=false
    local has_adc=false
    local sa_file=""

    if check_gcloud_auth; then
        has_gcloud_auth=true
    else
        print_color $YELLOW "○ No active gcloud authentication"
    fi

    if check_adc; then
        has_adc=true
    else
        print_color $YELLOW "○ No Application Default Credentials found"
    fi

    # Search for gcp-profile service account files
    print_color $BLUE "Searching for gcp-profile service account files..."

    local found_sa_files=()
    while IFS= read -r file; do
        if [[ -n "$file" ]]; then
            found_sa_files+=("$file")
        fi
    done < <(find_gcp_profile_sa)

    if [[ ${#found_sa_files[@]} -gt 0 ]]; then
        print_color $GREEN "Found ${#found_sa_files[@]} potential service account file(s):"

        local valid_files=()
        for file in "${found_sa_files[@]}"; do
            echo
            print_color $CYAN "Checking: $file"
            if validate_sa_file "$file"; then
                valid_files+=("$file")
                print_color $GREEN "✓ Valid service account file"
            else
                print_color $YELLOW "⚠ Invalid or incomplete service account file"
            fi
        done

        if [[ ${#valid_files[@]} -gt 0 ]]; then
            # Use the first valid file or prompt user to choose
            if [[ ${#valid_files[@]} -eq 1 ]]; then
                sa_file="${valid_files[0]}"
                print_color $GREEN "Using service account file: $sa_file"
            else
                echo
                print_color $BLUE "Multiple valid service account files found:"
                for i in "${!valid_files[@]}"; do
                    echo "  $((i+1)). ${valid_files[i]}"
                done

                while true; do
                    read -p "Select file to use (1-${#valid_files[@]}): " choice
                    if [[ "$choice" =~ ^[0-9]+$ ]] && [[ $choice -ge 1 ]] && [[ $choice -le ${#valid_files[@]} ]]; then
                        sa_file="${valid_files[$((choice-1))]}"
                        break
                    else
                        print_color $YELLOW "Invalid choice. Please enter a number between 1 and ${#valid_files[@]}."
                    fi
                done
            fi
        fi
    else
        print_color $YELLOW "○ No gcp-profile service account files found in common locations"
    fi

    # Prompt for authentication method
    echo
    print_color $BLUE "Choose authentication method:"

    local options=()
    local descriptions=()

    if [[ -n "$sa_file" ]]; then
        options+=("service_account")
        descriptions+=("Use found gcp-profile service account file")
    fi

    if [[ "$has_adc" == "true" ]]; then
        options+=("use_existing_adc")
        descriptions+=("Use existing Application Default Credentials")
    fi

    options+=("setup_adc")
    descriptions+=("Setup new Application Default Credentials")

    options+=("manual_sa")
    descriptions+=("Provide service account file path manually")

    for i in "${!options[@]}"; do
        echo "  $((i+1)). ${descriptions[i]}"
    done

    while true; do
        read -p "Select option (1-${#options[@]}): " choice
        if [[ "$choice" =~ ^[0-9]+$ ]] && [[ $choice -ge 1 ]] && [[ $choice -le ${#options[@]} ]]; then
            selected_option="${options[$((choice-1))]}"
            break
        else
            print_color $YELLOW "Invalid choice. Please enter a number between 1 and ${#options[@]}."
        fi
    done

    # Execute selected authentication method
    echo
    case "$selected_option" in
        "service_account")
            setup_service_account_auth "$sa_file"
            ;;
        "use_existing_adc")
            echo "GCP_AUTH_METHOD=\"application_default\"" >> "$ENV_FILE"
            print_color $GREEN "✓ Configured to use existing Application Default Credentials"
            ;;
        "setup_adc")
            setup_adc_auth
            ;;
        "manual_sa")
            while true; do
                read -p "Enter path to service account JSON file: " manual_sa_file
                if [[ -f "$manual_sa_file" ]]; then
                    if validate_sa_file "$manual_sa_file"; then
                        setup_service_account_auth "$manual_sa_file"
                        break
                    else
                        print_color $RED "Invalid service account file. Please try again."
                    fi
                else
                    print_color $RED "File not found: $manual_sa_file"
                fi
            done
            ;;
    esac

    # Test the authentication setup
    echo
    if test_authentication; then
        print_header "Setup Complete!"
        print_color $GREEN "GCP authentication has been configured successfully."
        echo
        print_color $GREEN "Next steps:"
        print_color $GREEN "  1. Source the environment: source .env"
        print_color $GREEN "  2. Install dependencies: uv pip install -r requirements.txt"
        print_color $GREEN "  3. Test your setup: python3 -c 'from src.gcp import GCPConfig; print(GCPConfig.from_env())'"
        echo

        if [[ -f "$ENV_FILE" ]]; then
            print_color $BLUE "Environment variables written to: $ENV_FILE"
            print_color $YELLOW "Remember to source this file: source .env"
        fi
    else
        print_header "Setup Issues"
        print_color $YELLOW "Authentication was configured but testing failed."
        print_color $YELLOW "Please check the following:"
        print_color $YELLOW "  1. Install Python dependencies: uv pip install -r requirements.txt"
        print_color $YELLOW "  2. Check that the service account has proper permissions"
        print_color $YELLOW "  3. Verify the project ID is correct"
        echo
        print_color $BLUE "Configuration files created:"
        if [[ -f "$ENV_FILE" ]]; then
            print_color $BLUE "  - Environment: $ENV_FILE"
        fi
        if [[ -f "$CONFIG_DIR/gcp-profile-sa.json" ]]; then
            print_color $BLUE "  - Service Account: $CONFIG_DIR/gcp-profile-sa.json"
        fi
    fi
}

# Execute main function
main "$@"
