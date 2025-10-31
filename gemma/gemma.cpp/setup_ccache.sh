#!/bin/bash
# CCCache Setup Script for Gemma.cpp Windows Build Optimization

# Color output functions
print_success() {
    echo -e "\033[32m✓ $1\033[0m"
}

print_info() {
    echo -e "\033[34mℹ $1\033[0m"
}

print_warning() {
    echo -e "\033[33m⚠ $1\033[0m"
}

print_error() {
    echo -e "\033[31m✗ $1\033[0m"
}

# Check if ccache is available
check_ccache() {
    if command -v ccache &> /dev/null; then
        print_success "CCCache is already installed: $(which ccache)"
        ccache --version | head -1
        return 0
    else
        print_warning "CCCache not found"
        return 1
    fi
}

# Configure ccache for optimal C++ builds
configure_ccache() {
    print_info "Configuring ccache for optimal performance..."

    # Set maximum cache size to 5GB (good for large C++ projects)
    ccache --set-config max_size=5G
    print_info "Set max cache size to 5GB"

    # Enable compression to save disk space
    ccache --set-config compression=true
    ccache --set-config compression_level=6
    print_info "Enabled compression (level 6)"

    # Set base directory for better cache hits
    local base_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
    ccache --set-config base_dir="$base_dir"
    print_info "Set base directory to: $base_dir"

    # Enable hash_dir for better cache hits with absolute paths
    ccache --set-config hash_dir=true
    print_info "Enabled hash_dir for absolute path handling"

    # Set compiler check to content (more reliable than mtime)
    ccache --set-config compiler_check=content
    print_info "Set compiler check to content-based"

    # Disable direct mode for better compatibility with MSVC
    ccache --set-config direct_mode=false
    print_info "Disabled direct mode for MSVC compatibility"

    # Set reasonable umask
    ccache --set-config umask=002
    print_info "Set umask to 002"

    print_success "CCCache configuration completed"
}

# Display current ccache configuration
show_config() {
    print_info "Current ccache configuration:"
    echo "----------------------------------------"
    ccache --show-config | grep -E "(max_size|compression|base_dir|hash_dir|compiler_check|direct_mode|umask|cache_dir)" | sort
    echo "----------------------------------------"
}

# Display ccache statistics
show_stats() {
    print_info "CCCache statistics:"
    echo "----------------------------------------"
    ccache --show-stats
    echo "----------------------------------------"
}

# Clear ccache statistics
clear_stats() {
    print_info "Clearing ccache statistics..."
    ccache --zero-stats
    print_success "Statistics cleared"
}

# Clear ccache
clear_cache() {
    print_warning "This will clear the entire ccache!"
    read -p "Are you sure? (y/N): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        print_info "Clearing ccache..."
        ccache --clear
        print_success "Cache cleared"
    else
        print_info "Cache clear cancelled"
    fi
}

# Create .ccache configuration file in home directory
create_config_file() {
    local config_file="$HOME/.ccache/ccache.conf"
    local config_dir="$(dirname "$config_file")"

    # Create directory if it doesn't exist
    if [ ! -d "$config_dir" ]; then
        mkdir -p "$config_dir"
        print_info "Created ccache config directory: $config_dir"
    fi

    # Create configuration file
    cat > "$config_file" << 'EOF'
# CCCache Configuration for Gemma.cpp Windows Build
# Optimized for large C++ projects with Windows/MSVC

# Cache size (5GB should be sufficient for most builds)
max_size = 5G

# Enable compression to save disk space
compression = true
compression_level = 6

# Use content-based compiler check for reliability
compiler_check = content

# Enable hash_dir for better absolute path handling
hash_dir = true

# Disable direct mode for MSVC compatibility
direct_mode = false

# Set reasonable umask
umask = 002

# Sloppiness settings for better compatibility
# (time_macros helps with Windows builds)
sloppiness = time_macros,include_file_mtime,include_file_ctime,file_stat_matches

# Debug settings (uncomment for troubleshooting)
# debug = true
# log_file = ccache.log
EOF

    print_success "Created ccache configuration file: $config_file"
}

# Install ccache (if not present)
install_ccache() {
    print_info "CCCache installation options:"
    echo "1. Using Chocolatey (recommended for Windows):"
    echo "   choco install ccache"
    echo ""
    echo "2. Using Scoop:"
    echo "   scoop install ccache"
    echo ""
    echo "3. Manual download:"
    echo "   https://ccache.dev/download.html"
    echo ""
    echo "4. WSL/Linux package manager:"
    echo "   sudo apt-get install ccache  # Ubuntu/Debian"
    echo "   sudo yum install ccache      # CentOS/RHEL"
    echo ""
    print_warning "Please install ccache and run this script again."
}

# Test ccache with a simple compilation
test_ccache() {
    if ! command -v ccache &> /dev/null; then
        print_error "CCCache not available for testing"
        return 1
    fi

    print_info "Testing ccache with simple compilation..."

    # Create a temporary test file
    local test_file="/tmp/ccache_test_$$.cpp"
    cat > "$test_file" << 'EOF'
#include <iostream>
int main() {
    std::cout << "CCCache test compilation" << std::endl;
    return 0;
}
EOF

    # Clear stats for clean test
    ccache --zero-stats

    # Try to compile with ccache (if g++ is available)
    if command -v g++ &> /dev/null; then
        print_info "Compiling test file with g++..."
        ccache g++ -o "/tmp/ccache_test_$$" "$test_file" 2>/dev/null
        if [ $? -eq 0 ]; then
            print_success "First compilation successful"

            # Compile again to test cache hit
            ccache g++ -o "/tmp/ccache_test_$$_2" "$test_file" 2>/dev/null
            if [ $? -eq 0 ]; then
                print_success "Second compilation successful"

                # Show stats
                local cache_hits=$(ccache --show-stats | grep "cache hit" | head -1)
                if [ -n "$cache_hits" ]; then
                    print_info "Cache statistics: $cache_hits"
                fi
            fi
        fi

        # Cleanup
        rm -f "/tmp/ccache_test_$$" "/tmp/ccache_test_$$_2"
    fi

    rm -f "$test_file"
    print_success "CCCache test completed"
}

# Main menu function
show_menu() {
    echo "CCCache Setup and Management for Gemma.cpp"
    echo "==========================================="
    echo "1. Check ccache installation"
    echo "2. Configure ccache"
    echo "3. Show configuration"
    echo "4. Show statistics"
    echo "5. Clear statistics"
    echo "6. Clear cache"
    echo "7. Create config file"
    echo "8. Test ccache"
    echo "9. Install help"
    echo "0. Exit"
    echo ""
}

# Main execution
main() {
    if [ $# -eq 0 ]; then
        # Interactive mode
        while true; do
            show_menu
            read -p "Choose an option (0-9): " choice
            echo ""

            case $choice in
                1)
                    check_ccache
                    ;;
                2)
                    if check_ccache; then
                        configure_ccache
                    else
                        print_error "CCCache not found. Please install it first."
                    fi
                    ;;
                3)
                    if check_ccache; then
                        show_config
                    else
                        print_error "CCCache not found."
                    fi
                    ;;
                4)
                    if check_ccache; then
                        show_stats
                    else
                        print_error "CCCache not found."
                    fi
                    ;;
                5)
                    if check_ccache; then
                        clear_stats
                    else
                        print_error "CCCache not found."
                    fi
                    ;;
                6)
                    if check_ccache; then
                        clear_cache
                    else
                        print_error "CCCache not found."
                    fi
                    ;;
                7)
                    create_config_file
                    ;;
                8)
                    test_ccache
                    ;;
                9)
                    install_ccache
                    ;;
                0)
                    print_info "Goodbye!"
                    exit 0
                    ;;
                *)
                    print_error "Invalid option. Please choose 0-9."
                    ;;
            esac
            echo ""
            read -p "Press Enter to continue..."
            echo ""
        done
    else
        # Command line mode
        case $1 in
            check)
                check_ccache
                ;;
            configure)
                if check_ccache; then
                    configure_ccache
                else
                    print_error "CCCache not found. Please install it first."
                    exit 1
                fi
                ;;
            config)
                if check_ccache; then
                    show_config
                else
                    print_error "CCCache not found."
                    exit 1
                fi
                ;;
            stats)
                if check_ccache; then
                    show_stats
                else
                    print_error "CCCache not found."
                    exit 1
                fi
                ;;
            clear-stats)
                if check_ccache; then
                    clear_stats
                else
                    print_error "CCCache not found."
                    exit 1
                fi
                ;;
            clear-cache)
                if check_ccache; then
                    clear_cache
                else
                    print_error "CCCache not found."
                    exit 1
                fi
                ;;
            create-config)
                create_config_file
                ;;
            test)
                test_ccache
                ;;
            install-help)
                install_ccache
                ;;
            auto-setup)
                # Automatic setup for CI/scripts
                if check_ccache; then
                    configure_ccache
                    create_config_file
                    print_success "Auto-setup completed"
                else
                    print_error "CCCache not found. Auto-setup failed."
                    install_ccache
                    exit 1
                fi
                ;;
            *)
                echo "Usage: $0 [check|configure|config|stats|clear-stats|clear-cache|create-config|test|install-help|auto-setup]"
                echo "Run without arguments for interactive mode."
                exit 1
                ;;
        esac
    fi
}

# Run main function with all arguments
main "$@"