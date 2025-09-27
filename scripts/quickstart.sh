#!/bin/bash

# Spyglass Quickstart Script
# One-command setup for Spyglass installation
#
# Usage:
#   ./quickstart.sh [OPTIONS]
#
# Options:
#   --minimal     : Core functionality only (default)
#   --full        : All optional dependencies
#   --pipeline=X  : Specific pipeline (dlc|moseq-cpu|moseq-gpu|decoding|lfp)
#   --no-database : Skip database setup
#   --no-validate : Skip validation after setup
#   --help        : Show this help message

set -e  # Exit on error

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color
BOLD='\033[1m'

# Default options
INSTALL_TYPE="minimal"
PIPELINE=""
SETUP_DATABASE=true
RUN_VALIDATION=true
CONDA_CMD=""
BASE_DIR="$HOME/spyglass_data"

# Script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Function to print colored output
print_color() {
    local color=$1
    shift
    echo -e "${color}$*${NC}"
}

print_header() {
    echo
    print_color "$CYAN" "=========================================="
    print_color "$CYAN" "$BOLD$1"
    print_color "$CYAN" "=========================================="
    echo
}

print_success() {
    print_color "$GREEN" "✓ $1"
}

print_warning() {
    print_color "$YELLOW" "⚠ $1"
}

print_error() {
    print_color "$RED" "✗ $1"
}

print_info() {
    print_color "$BLUE" "ℹ $1"
}

# Show help
show_help() {
    cat << EOF
Spyglass Quickstart Script

This script provides a streamlined setup process for Spyglass, guiding you
through environment creation, package installation, and configuration.

Usage:
    ./quickstart.sh [OPTIONS]

Options:
    --minimal       Install core dependencies only (default)
    --full          Install all optional dependencies
    --pipeline=X    Install specific pipeline dependencies:
                    - dlc: DeepLabCut for position tracking
                    - moseq-cpu: Keypoint-Moseq (CPU version)
                    - moseq-gpu: Keypoint-Moseq (GPU version)
                    - lfp: Local Field Potential analysis
                    - decoding: Neural decoding with JAX
    --no-database   Skip database setup
    --no-validate   Skip validation after setup
    --base-dir=PATH Set base directory for data (default: ~/spyglass_data)
    --help          Show this help message

Examples:
    # Minimal installation with validation
    ./quickstart.sh

    # Full installation with all dependencies
    ./quickstart.sh --full

    # Install for DeepLabCut pipeline
    ./quickstart.sh --pipeline=dlc

    # Custom base directory
    ./quickstart.sh --base-dir=/data/spyglass

EOF
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --minimal)
            INSTALL_TYPE="minimal"
            shift
            ;;
        --full)
            INSTALL_TYPE="full"
            shift
            ;;
        --pipeline=*)
            PIPELINE="${1#*=}"
            shift
            ;;
        --no-database)
            SETUP_DATABASE=false
            shift
            ;;
        --no-validate)
            RUN_VALIDATION=false
            shift
            ;;
        --base-dir=*)
            BASE_DIR="${1#*=}"
            shift
            ;;
        --help)
            show_help
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Detect operating system
detect_os() {
    print_header "System Detection"

    OS=$(uname -s)
    ARCH=$(uname -m)

    case "$OS" in
        Darwin)
            OS_NAME="macOS"
            print_success "Operating System: macOS"
            if [[ "$ARCH" == "arm64" ]]; then
                print_success "Architecture: Apple Silicon (M1/M2)"
                IS_M1=true
            else
                print_success "Architecture: Intel x86_64"
                IS_M1=false
            fi
            ;;
        Linux)
            OS_NAME="Linux"
            print_success "Operating System: Linux"
            print_success "Architecture: $ARCH"
            IS_M1=false
            ;;
        *)
            print_error "Unsupported operating system: $OS"
            print_info "Spyglass officially supports macOS and Linux only"
            exit 1
            ;;
    esac
}

# Check Python version
check_python() {
    print_header "Python Check"

    if command -v python3 &> /dev/null; then
        PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
        PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
        PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

        if [[ $PYTHON_MAJOR -ge 3 ]] && [[ $PYTHON_MINOR -ge 9 ]]; then
            print_success "Python $PYTHON_VERSION found"
        else
            print_warning "Python $PYTHON_VERSION found, but Python >= 3.9 is required"
            print_info "The conda environment will install the correct version"
        fi
    else
        print_warning "Python 3 not found in PATH"
        print_info "The conda environment will install Python"
    fi
}

# Check for conda/mamba
check_conda() {
    print_header "Package Manager Check"

    if command -v mamba &> /dev/null; then
        CONDA_CMD="mamba"
        print_success "Found mamba (recommended)"
    elif command -v conda &> /dev/null; then
        CONDA_CMD="conda"
        print_success "Found conda"
        print_info "Consider installing mamba for faster environment creation:"
        print_info "  conda install -n base -c conda-forge mamba"
    else
        print_error "Neither mamba nor conda found"
        print_info "Please install miniforge or miniconda:"
        print_info "  https://github.com/conda-forge/miniforge#install"
        exit 1
    fi

    # Show conda info
    CONDA_VERSION=$($CONDA_CMD --version 2>&1)
    print_info "Version: $CONDA_VERSION"
}

# Select environment file based on options
select_environment() {
    print_header "Environment Selection"

    local env_file=""

    if [[ -n "$PIPELINE" ]]; then
        case "$PIPELINE" in
            dlc)
                env_file="environment_dlc.yml"
                print_info "Selected: DeepLabCut pipeline environment"
                ;;
            moseq-cpu)
                env_file="environment_moseq_cpu.yml"
                print_info "Selected: Keypoint-Moseq CPU environment"
                ;;
            moseq-gpu)
                env_file="environment_moseq_gpu.yml"
                print_info "Selected: Keypoint-Moseq GPU environment"
                ;;
            lfp|decoding)
                env_file="environment.yml"
                print_info "Selected: Standard environment (will add $PIPELINE dependencies)"
                ;;
            *)
                print_error "Unknown pipeline: $PIPELINE"
                print_info "Valid options: dlc, moseq-cpu, moseq-gpu, lfp, decoding"
                exit 1
                ;;
        esac
    elif [[ "$INSTALL_TYPE" == "full" ]]; then
        env_file="environment.yml"
        print_info "Selected: Standard environment (will add all optional dependencies)"
    else
        env_file="environment.yml"
        print_info "Selected: Standard environment (minimal)"
    fi

    # Check if environment file exists
    if [[ ! -f "$REPO_DIR/$env_file" ]]; then
        print_error "Environment file not found: $REPO_DIR/$env_file"
        exit 1
    fi

    echo "$env_file"
}

# Create conda environment
create_environment() {
    local env_file=$1
    print_header "Creating Conda Environment"

    ENV_NAME="spyglass"

    # Check if environment already exists
    if $CONDA_CMD env list | grep -q "^$ENV_NAME "; then
        print_warning "Environment '$ENV_NAME' already exists"
        read -p "Do you want to update it? (y/N): " -n 1 -r
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            print_info "Keeping existing environment"
            return
        fi
        print_info "Updating existing environment..."
        $CONDA_CMD env update -f "$REPO_DIR/$env_file" -n $ENV_NAME
    else
        print_info "Creating new environment '$ENV_NAME'..."
        print_info "This may take 5-10 minutes..."
        $CONDA_CMD env create -f "$REPO_DIR/$env_file" -n $ENV_NAME
    fi

    print_success "Environment created/updated successfully"
}

# Install additional dependencies
install_additional_deps() {
    print_header "Installing Additional Dependencies"

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate spyglass

    # Install Spyglass in development mode
    print_info "Installing Spyglass in development mode..."
    pip install -e "$REPO_DIR"

    # Install pipeline-specific dependencies
    if [[ "$PIPELINE" == "lfp" ]]; then
        print_info "Installing LFP dependencies..."
        if [[ "$IS_M1" == true ]]; then
            print_info "Detected M1 Mac, installing pyfftw via conda first..."
            conda install -c conda-forge pyfftw -y
        fi
        pip install ghostipy
    elif [[ "$PIPELINE" == "decoding" ]]; then
        print_info "Installing decoding dependencies..."
        print_info "Please refer to JAX installation guide for GPU support:"
        print_info "https://jax.readthedocs.io/en/latest/installation.html"
    fi

    # Install all optional dependencies if --full
    if [[ "$INSTALL_TYPE" == "full" ]]; then
        print_info "Installing all optional dependencies..."
        pip install spikeinterface[full,widgets]
        pip install mountainsort4

        if [[ "$IS_M1" == true ]]; then
            conda install -c conda-forge pyfftw -y
        fi
        pip install ghostipy

        print_warning "Some dependencies (DLC, JAX) require separate environment files"
        print_info "Use --pipeline=dlc or --pipeline=moseq-gpu for those"
    fi

    print_success "Additional dependencies installed"
}

# Setup database
setup_database() {
    if [[ "$SETUP_DATABASE" == false ]]; then
        print_info "Skipping database setup (--no-database specified)"
        return
    fi

    print_header "Database Setup"

    echo "Choose database setup option:"
    echo "1) Local Docker database (recommended for beginners)"
    echo "2) Connect to existing database"
    echo "3) Skip database setup"

    read -p "Enter choice (1-3): " -n 1 -r
    echo

    case $REPLY in
        1)
            setup_docker_database
            ;;
        2)
            setup_existing_database
            ;;
        3)
            print_info "Skipping database setup"
            print_warning "You'll need to configure the database manually later"
            ;;
        *)
            print_error "Invalid choice"
            exit 1
            ;;
    esac
}

# Setup Docker database
setup_docker_database() {
    print_info "Setting up local Docker database..."

    # Check if Docker is installed
    if ! command -v docker &> /dev/null; then
        print_error "Docker is not installed"
        print_info "Please install Docker from: https://docs.docker.com/engine/install/"
        exit 1
    fi

    # Check if Docker daemon is running
    if ! docker info &> /dev/null; then
        print_error "Docker daemon is not running"
        print_info "Please start Docker and try again"
        exit 1
    fi

    # Pull and run MySQL container
    print_info "Setting up MySQL container..."
    docker pull datajoint/mysql:8.0

    # Check if container already exists
    if docker ps -a | grep -q spyglass-db; then
        print_warning "Container 'spyglass-db' already exists"
        docker start spyglass-db
    else
        docker run -d --name spyglass-db \
            -p 3306:3306 \
            -e MYSQL_ROOT_PASSWORD=tutorial \
            datajoint/mysql:8.0
    fi

    print_success "Docker database started"

    # Create config file
    create_config "localhost" "root" "tutorial" "3306"
}

# Setup connection to existing database
setup_existing_database() {
    print_info "Configuring connection to existing database..."

    read -p "Database host: " db_host
    read -p "Database port (3306): " db_port
    db_port=${db_port:-3306}
    read -p "Database user: " db_user
    read -s -p "Database password: " db_password
    echo

    create_config "$db_host" "$db_user" "$db_password" "$db_port"
}

# Create DataJoint config file
create_config() {
    local host=$1
    local user=$2
    local password=$3
    local port=$4

    print_info "Creating configuration file..."

    # Create base directory structure
    mkdir -p "$BASE_DIR"/{raw,analysis,recording,sorting,tmp,video,waveforms}

    # Create config file
    cat > "$REPO_DIR/dj_local_conf.json" << EOF
{
    "database.host": "$host",
    "database.port": $port,
    "database.user": "$user",
    "database.password": "$password",
    "database.reconnect": true,
    "database.use_tls": false,
    "stores": {
        "raw": {
            "protocol": "file",
            "location": "$BASE_DIR/raw"
        },
        "analysis": {
            "protocol": "file",
            "location": "$BASE_DIR/analysis"
        }
    },
    "custom": {
        "spyglass_dirs": {
            "base_dir": "$BASE_DIR",
            "raw_dir": "$BASE_DIR/raw",
            "analysis_dir": "$BASE_DIR/analysis",
            "recording_dir": "$BASE_DIR/recording",
            "sorting_dir": "$BASE_DIR/sorting",
            "temp_dir": "$BASE_DIR/tmp",
            "video_dir": "$BASE_DIR/video",
            "waveforms_dir": "$BASE_DIR/waveforms"
        }
    }
}
EOF

    print_success "Configuration file created at: $REPO_DIR/dj_local_conf.json"
    print_success "Data directories created at: $BASE_DIR"
}

# Run validation
run_validation() {
    if [[ "$RUN_VALIDATION" == false ]]; then
        print_info "Skipping validation (--no-validate specified)"
        return
    fi

    print_header "Running Validation"

    # Activate environment
    eval "$(conda shell.bash hook)"
    conda activate spyglass

    # Run validation script
    if [[ -f "$SCRIPT_DIR/validate_spyglass.py" ]]; then
        print_info "Running validation checks..."
        python "$SCRIPT_DIR/validate_spyglass.py" -v

        if [[ $? -eq 0 ]]; then
            print_success "All validation checks passed!"
        elif [[ $? -eq 1 ]]; then
            print_warning "Validation passed with warnings"
            print_info "Review the warnings above if you need specific features"
        else
            print_error "Validation failed"
            print_info "Please review the errors above and fix any issues"
        fi
    else
        print_error "Validation script not found"
    fi
}

# Print final instructions
print_summary() {
    print_header "Setup Complete!"

    echo "Next steps:"
    echo
    echo "1. Activate the Spyglass environment:"
    print_color "$GREEN" "   conda activate spyglass"
    echo
    echo "2. Start with the tutorials:"
    print_color "$GREEN" "   cd $REPO_DIR/notebooks"
    print_color "$GREEN" "   jupyter notebook 01_Concepts.ipynb"
    echo
    echo "3. For help and documentation:"
    print_color "$BLUE" "   Documentation: https://lorenfranklab.github.io/spyglass/"
    print_color "$BLUE" "   GitHub Issues: https://github.com/LorenFrankLab/spyglass/issues"
    echo

    if [[ "$SETUP_DATABASE" == false ]]; then
        print_warning "Remember to configure your database connection"
        print_info "See: https://lorenfranklab.github.io/spyglass/latest/notebooks/00_Setup/"
    fi
}

# Main execution
main() {
    print_color "$CYAN" "$BOLD"
    echo "╔═══════════════════════════════════════╗"
    echo "║     Spyglass Quickstart Installer    ║"
    echo "╚═══════════════════════════════════════╝"
    print_color "$NC" ""

    # Run setup steps
    detect_os
    check_python
    check_conda

    # Select and create environment
    ENV_FILE=$(select_environment)
    create_environment "$ENV_FILE"

    # Install additional dependencies
    install_additional_deps

    # Setup database
    setup_database

    # Run validation
    run_validation

    # Print summary
    print_summary
}

# Run main function
main