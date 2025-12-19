#!/usr/bin/env bash
# Frank Lab Spyglass Setup Script
#
# This script pre-configures lab-wide settings for Frank Lab members,
# so they only need to enter their personal database credentials.
#
# Supports two scenarios:
#   1. On Frank Lab server (stelmo, nimbus) - uses shared /stelmo/nwb directory
#   2. On personal laptop - uses local ~/spyglass_data directory
#
# Usage:
#   ./scripts/setup_franklab.sh              # Interactive (auto-detects scenario)
#   ./scripts/setup_franklab.sh --user alice # Non-interactive with username
#   ./scripts/setup_franklab.sh --local      # Force local laptop mode
#
# Lab-wide settings (pre-configured):
#   - Database host: lmf-db.cin.ucsf.edu
#   - Database port: 3306
#   - TLS: enabled (remote server)
#
# User-specific settings (prompted):
#   - Database username
#   - Database password (entered securely, then prompted to change)

set -e  # Exit on error

# ============================================================================
# Frank Lab Configuration
# ============================================================================
FRANKLAB_DB_HOST="lmf-db.cin.ucsf.edu"
FRANKLAB_DB_PORT="3306"
FRANKLAB_SERVER_DATA_DIR="/stelmo/nwb"
FRANKLAB_LOCAL_DATA_DIR="$HOME/spyglass_data"

# ============================================================================
# Script setup
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_step() {
    echo -e "${BLUE}▶${NC} $1"
}

print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

# ============================================================================
# Parse arguments
# ============================================================================
DB_USER=""
INSTALL_TYPE="--minimal"
FORCE_LOCAL=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --user|-u)
            DB_USER="$2"
            shift 2
            ;;
        --full)
            INSTALL_TYPE="--full"
            shift
            ;;
        --local|-l)
            FORCE_LOCAL=true
            shift
            ;;
        --help|-h)
            echo "Frank Lab Spyglass Setup"
            echo ""
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --user, -u USERNAME   Database username (default: prompts)"
            echo "  --full                Install full dependencies (default: minimal)"
            echo "  --local, -l           Force local mode (laptop, not on server)"
            echo "  --help, -h            Show this help message"
            echo ""
            echo "Scenarios:"
            echo "  On Frank Lab server:  Data stored in $FRANKLAB_SERVER_DATA_DIR"
            echo "  On personal laptop:   Data stored in $FRANKLAB_LOCAL_DATA_DIR"
            echo ""
            echo "Lab-wide settings (pre-configured):"
            echo "  Database host: $FRANKLAB_DB_HOST"
            echo "  Database port: $FRANKLAB_DB_PORT"
            echo ""
            echo "Examples:"
            echo "  $0                    # Auto-detect server vs laptop"
            echo "  $0 --user alice       # Non-interactive with username"
            echo "  $0 --local            # Force laptop mode"
            exit 0
            ;;
        *)
            print_error "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ============================================================================
# Detect scenario: server vs laptop
# ============================================================================
echo ""
echo "============================================================"
echo "  Frank Lab Spyglass Setup"
echo "============================================================"
echo ""

# Check we're in the right directory
if [[ ! -f "$REPO_DIR/scripts/install.py" ]]; then
    print_error "Cannot find install.py. Please run from the spyglass repository."
    echo "  Expected: $REPO_DIR/scripts/install.py"
    exit 1
fi

# Detect if on Frank Lab server or personal laptop
if [[ "$FORCE_LOCAL" == true ]]; then
    ON_SERVER=false
    print_step "Forced local mode (laptop)"
elif [[ -d "$FRANKLAB_SERVER_DATA_DIR" ]]; then
    ON_SERVER=true
    print_success "Detected: Frank Lab server (stelmo/nimbus)"
    echo "  Data directory: $FRANKLAB_SERVER_DATA_DIR"
else
    ON_SERVER=false
    print_step "Detected: Personal laptop (remote connection to database)"
    echo "  Data directory: $FRANKLAB_LOCAL_DATA_DIR"
fi

# Set the appropriate base directory
if [[ "$ON_SERVER" == true ]]; then
    BASE_DIR="$FRANKLAB_SERVER_DATA_DIR"
else
    BASE_DIR="$FRANKLAB_LOCAL_DATA_DIR"
fi

echo ""

# ============================================================================
# Get username
# ============================================================================
if [[ -z "$DB_USER" ]]; then
    print_step "Enter your Frank Lab database username"
    echo "  (This was provided by your lab admin)"
    echo ""
    read -p "Username: " DB_USER

    if [[ -z "$DB_USER" ]]; then
        print_error "Username cannot be empty"
        exit 1
    fi
fi

print_success "Database user: $DB_USER"

# ============================================================================
# Get password (securely)
# ============================================================================
echo ""
print_step "Enter your database password"
echo "  (You'll be prompted to change this after connecting)"
echo ""
read -s -p "Password: " DB_PASSWORD
echo ""  # New line after hidden input

if [[ -z "$DB_PASSWORD" ]]; then
    print_error "Password cannot be empty"
    exit 1
fi

print_success "Password entered (hidden)"

# ============================================================================
# Show configuration summary
# ============================================================================
echo ""
echo "Configuration:"
echo "  Database:  $FRANKLAB_DB_HOST:$FRANKLAB_DB_PORT"
echo "  User:      $DB_USER"
echo "  Data dir:  $BASE_DIR"
echo "  Install:   ${INSTALL_TYPE#--} dependencies"
if [[ "$ON_SERVER" == true ]]; then
    echo "  Mode:      On Frank Lab server (shared data)"
else
    echo "  Mode:      Personal laptop (local data)"
fi
echo ""

read -p "Continue with installation? [Y/n]: " CONFIRM
CONFIRM=${CONFIRM:-y}

if [[ ! "$CONFIRM" =~ ^[Yy] ]]; then
    echo "Installation cancelled."
    exit 0
fi

# ============================================================================
# Set environment and run installer
# ============================================================================
echo ""
print_step "Starting Spyglass installation..."
echo ""

# Export lab-wide settings as environment variables
export SPYGLASS_BASE_DIR="$BASE_DIR"
# Pass password via environment variable to avoid exposure in process listings
export SPYGLASS_DB_PASSWORD="$DB_PASSWORD"

# Run the installer with Frank Lab settings
cd "$REPO_DIR"
python scripts/install.py \
    $INSTALL_TYPE \
    --remote \
    --db-host "$FRANKLAB_DB_HOST" \
    --db-port "$FRANKLAB_DB_PORT" \
    --db-user "$DB_USER" \
    --base-dir "$BASE_DIR"

# ============================================================================
# Post-installation notes
# ============================================================================
echo ""
echo "============================================================"
echo "  Frank Lab Setup Complete"
echo "============================================================"
echo ""
echo "Next steps:"
echo "  1. Activate environment: conda activate spyglass"
echo "  2. Test connection: python -c \"import datajoint as dj; dj.conn()\""
echo "  3. Start tutorials: jupyter notebook notebooks/"
echo ""
echo "Your data will be stored in: $BASE_DIR"

if [[ "$ON_SERVER" == false ]]; then
    echo ""
    print_warning "Note: You're on a personal laptop"
    echo "  - Raw NWB files need to be copied locally or accessed via network"
    echo "  - For large datasets, consider working on a Frank Lab server"
    echo "  - SSH to stelmo/nimbus for shared data access"
fi

echo ""
echo "Questions? Ask in #spyglass on Slack or GitHub Discussions"
echo ""
