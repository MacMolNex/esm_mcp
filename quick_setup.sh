#!/bin/bash
#===============================================================================
# ESM MCP Quick Setup Script
#===============================================================================
# This script sets up the complete environment for ESM MCP server.
#
# After cloning the repository, run this script to set everything up:
#   cd esm_mcp
#   bash quick_setup.sh
#
# Once setup is complete, register in Claude Code with the config shown at the end.
#
# Options:
#   --skip-env        Skip conda environment creation
#   --skip-repo       Skip cloning ESM repository
#   --help            Show this help message
#===============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${SCRIPT_DIR}/env"
ENV_ESMFOLD_DIR="${SCRIPT_DIR}/env_esmfold"
PYTHON_VERSION="3.10"
REPO_DIR="${SCRIPT_DIR}/repo"
ESM_REPO="https://github.com/facebookresearch/esm.git"

# Print banner
echo -e "${BLUE}"
echo "=============================================="
echo "        ESM MCP Quick Setup Script           "
echo "=============================================="
echo -e "${NC}"

# Helper functions
info() { echo -e "${BLUE}[INFO]${NC} $1"; }
success() { echo -e "${GREEN}[SUCCESS]${NC} $1"; }
warn() { echo -e "${YELLOW}[WARNING]${NC} $1"; }
error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check for conda/mamba
check_conda() {
    if command -v mamba &> /dev/null; then
        CONDA_CMD="mamba"
        info "Using mamba (faster package resolution)"
    elif command -v conda &> /dev/null; then
        CONDA_CMD="conda"
        info "Using conda"
    else
        error "Neither conda nor mamba found. Please install Miniconda or Mambaforge first."
        exit 1
    fi
}

# Parse arguments
SKIP_ENV=false
SKIP_REPO=false
SKIP_ESMFOLD=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-env) SKIP_ENV=true; shift ;;
        --skip-repo) SKIP_REPO=true; shift ;;
        --skip-esmfold) SKIP_ESMFOLD=true; shift ;;
        -h|--help)
            echo "Usage: ./quick_setup.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-env        Skip conda environment creation"
            echo "  --skip-repo       Skip cloning ESM repository"
            echo "  --skip-esmfold    Skip ESMFold environment (not needed for embeddings)"
            echo "  -h, --help        Show this help message"
            exit 0
            ;;
        *) warn "Unknown option: $1"; shift ;;
    esac
done

# Check prerequisites
info "Checking prerequisites..."
check_conda

if ! command -v git &> /dev/null; then
    error "git is not installed. Please install git first."
    exit 1
fi
success "Prerequisites check passed"

# Step 1: Create main conda environment
echo ""
echo -e "${BLUE}Step 1: Setting up main conda environment${NC}"

# Fast path: use pre-packaged conda env from GitHub Releases
PACKED_ENV_URL="${PACKED_ENV_URL:-}"
PACKED_ENV_TAG="${PACKED_ENV_TAG:-envs-v1}"
PACKED_ENV_BASE="https://github.com/charlesxu90/ProteinMCP/releases/download/${PACKED_ENV_TAG}"

if [ "$SKIP_ENV" = true ]; then
    info "Skipping environment creation (--skip-env)"
elif [ -d "$ENV_DIR" ] && [ -f "$ENV_DIR/bin/python" ]; then
    info "Environment already exists at: $ENV_DIR"
elif [ "${USE_PACKED_ENVS:-}" = "1" ] || [ -n "$PACKED_ENV_URL" ]; then
    # Download and extract pre-packaged conda environment
    mkdir -p "$ENV_DIR"
    PACKED_OK=false

    if [ -n "$PACKED_ENV_URL" ]; then
        # User-specified URL: single file
        info "Downloading pre-packaged environment from ${PACKED_ENV_URL}..."
        if wget -qO- "$PACKED_ENV_URL" | tar xzf - -C "$ENV_DIR"; then
            PACKED_OK=true
        fi
    else
        # Try single file first, then split parts
        SINGLE_URL="${PACKED_ENV_BASE}/esm_mcp-env.tar.gz"
        info "Downloading pre-packaged environment from ${SINGLE_URL}..."
        if wget -qO- "$SINGLE_URL" | tar xzf - -C "$ENV_DIR" 2>/dev/null; then
            PACKED_OK=true
        else
            # Try split parts (part-aa, part-ab, ...)
            info "Single file not found, trying split parts..."
            TMPDIR_PARTS=$(mktemp -d)
            PART_OK=true
            for SUFFIX in aa ab ac ad ae; do
                PART_URL="${PACKED_ENV_BASE}/esm_mcp-env.tar.gz.part-${SUFFIX}"
                if wget -q -O "${TMPDIR_PARTS}/part-${SUFFIX}" "$PART_URL" 2>/dev/null; then
                    info "Downloaded part-${SUFFIX}"
                else
                    break
                fi
            done
            if ls "${TMPDIR_PARTS}"/part-* >/dev/null 2>&1; then
                info "Reassembling and extracting split archive..."
                if cat "${TMPDIR_PARTS}"/part-* | tar xzf - -C "$ENV_DIR"; then
                    PACKED_OK=true
                fi
            fi
            rm -rf "$TMPDIR_PARTS"
        fi
    fi

    if [ "$PACKED_OK" = true ]; then
        source "$ENV_DIR/bin/activate"
        conda-unpack 2>/dev/null || true
        success "Pre-packaged environment ready"
        SKIP_ENV=true
        SKIP_ESMFOLD=true  # Pre-packaged env doesn't include ESMFold
    else
        warn "Failed to download pre-packaged env, falling back to conda create..."
        rm -rf "$ENV_DIR"
        info "Creating conda environment with Python ${PYTHON_VERSION}..."
        $CONDA_CMD create -p "$ENV_DIR" python=${PYTHON_VERSION} pip -y
    fi
else
    info "Creating conda environment with Python ${PYTHON_VERSION}..."
    $CONDA_CMD create -p "$ENV_DIR" python=${PYTHON_VERSION} pip -y
fi

# Step 2: Clone repository
echo ""
echo -e "${BLUE}Step 2: Cloning ESM repository${NC}"

if [ "$SKIP_REPO" = true ]; then
    info "Skipping repository clone (--skip-repo)"
elif [ -d "$REPO_DIR/esm" ]; then
    info "ESM repository already exists"
else
    info "Cloning ESM repository..."
    mkdir -p "$REPO_DIR"
    git clone --depth 1 "$ESM_REPO" "$REPO_DIR/esm"
    success "Repository cloned"
fi

# Step 3: Install dependencies
echo ""
echo -e "${BLUE}Step 3: Installing dependencies${NC}"

if [ "$SKIP_ENV" = true ]; then
    info "Skipping dependency installation (--skip-env)"
else
    info "Installing MCP dependencies to main env..."
    "${ENV_DIR}/bin/pip" install fastmcp loguru

    # Create ESMFold environment if environment.yml exists (skip with --skip-esmfold)
    if [ "$SKIP_ESMFOLD" = true ]; then
        info "Skipping ESMFold environment (--skip-esmfold)"
    elif [ -f "$REPO_DIR/esm/environment.yml" ]; then
        echo ""
        echo -e "${BLUE}Step 3b: Setting up ESMFold environment${NC}"
        if [ -d "$ENV_ESMFOLD_DIR" ] && [ -f "$ENV_ESMFOLD_DIR/bin/python" ]; then
            info "ESMFold environment already exists"
        else
            info "Creating ESMFold environment from environment.yml..."
            $CONDA_CMD env create -f "$REPO_DIR/esm/environment.yml" -p "$ENV_ESMFOLD_DIR" || warn "ESMFold environment creation failed"
        fi

        if [ -d "$ENV_ESMFOLD_DIR" ]; then
            info "Installing ESM in editable mode..."
            "${ENV_ESMFOLD_DIR}/bin/pip" install -e "$REPO_DIR/esm/"
        fi
    fi

    info "Reinstalling fastmcp..."
    "${ENV_DIR}/bin/pip" install --ignore-installed fastmcp
    success "Dependencies installed"
fi

# Step 4: Verify installation
echo ""
echo -e "${BLUE}Step 4: Verifying installation${NC}"

"${ENV_DIR}/bin/python" -c "import fastmcp; import loguru; print('Core packages OK')" && success "Core packages verified" || error "Package verification failed"

# Print summary
echo ""
echo -e "${GREEN}=============================================="
echo "           Setup Complete!"
echo "==============================================${NC}"
echo ""
echo "Main Environment:    $ENV_DIR"
echo "ESMFold Environment: $ENV_ESMFOLD_DIR"
echo "Repository:          $REPO_DIR/esm"
echo ""
echo -e "${YELLOW}Claude Code Configuration:${NC}"
echo ""
cat << EOF
{
  "mcpServers": {
    "esm": {
      "command": "${ENV_DIR}/bin/python",
      "args": ["${SCRIPT_DIR}/src/server.py"]
    }
  }
}
EOF
echo ""
echo "To add to Claude Code:"
echo "  claude mcp add esm -- ${ENV_DIR}/bin/python ${SCRIPT_DIR}/src/server.py"
echo ""
