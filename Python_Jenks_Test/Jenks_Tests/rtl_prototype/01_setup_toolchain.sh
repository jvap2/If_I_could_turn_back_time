#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────
# GF4 RTL synthesis toolchain setup — no root/sudo required.
#
# Installs, into $HOME/eda:
#   - OSS CAD Suite (Yosys, Verilator, Icarus Verilog, OpenSTA, GTKWave, etc.)
#     as prebuilt static binaries — no apt/root needed.
#   - The open Nangate45 standard-cell library (liberty + LEF), pulled from
#     OpenROAD-flow-scripts via a sparse/shallow git checkout. This is the
#     same 45nm node your Timeloop/Accelergy model already targets, so the
#     synthesized area number is directly comparable, no node-scaling caveat.
#
# Usage:
#   chmod +x 01_setup_toolchain.sh
#   ./01_setup_toolchain.sh
#   source ~/eda/oss-cad-suite/environment   # do this in every new shell
# ─────────────────────────────────────────────────────────────────────────
set -euo pipefail

EDA_DIR="${EDA_DIR:-$HOME/eda}"
mkdir -p "$EDA_DIR"
cd "$EDA_DIR"

# ---- 1. Detect OS/arch and pick the right OSS CAD Suite release asset ----
UNAME_S="$(uname -s)"
UNAME_M="$(uname -m)"

case "$UNAME_S" in
  Linux)
    case "$UNAME_M" in
      x86_64)  PLATFORM="linux-x64" ;;
      aarch64) PLATFORM="linux-arm64" ;;
      *) echo "Unsupported Linux arch: $UNAME_M"; exit 1 ;;
    esac
    ;;
  Darwin)
    case "$UNAME_M" in
      x86_64) PLATFORM="darwin-x64" ;;
      arm64)  PLATFORM="darwin-arm64" ;;
      *) echo "Unsupported macOS arch: $UNAME_M"; exit 1 ;;
    esac
    ;;
  *)
    echo "Unsupported OS: $UNAME_S (this script targets Linux/macOS)"; exit 1
    ;;
esac

echo "Detected platform: $PLATFORM"

# ---- 2. Download OSS CAD Suite (pick latest release tag automatically) ----
if [ ! -d "$EDA_DIR/oss-cad-suite" ]; then
  echo "Fetching latest OSS CAD Suite release info..."
  LATEST_TAG=$(curl -s https://api.github.com/repos/YosysHQ/oss-cad-suite-build/releases/latest \
    | grep -oP '"tag_name":\s*"\K[^"]+')
  DATE_STR="${LATEST_TAG#build}"   # tags look like "2025-01-15"
  ASSET="oss-cad-suite-${PLATFORM}-${LATEST_TAG//-/}.tgz"
  URL="https://github.com/YosysHQ/oss-cad-suite-build/releases/download/${LATEST_TAG}/${ASSET}"

  echo "Downloading: $URL"
  if ! curl -fL -o oss-cad-suite.tgz "$URL"; then
    echo ""
    echo "Automatic asset-name resolution failed (GitHub sometimes changes the"
    echo "naming scheme). Go to the releases page, copy the .tgz link for your"
    echo "platform ($PLATFORM), and run:"
    echo "  curl -fL -o oss-cad-suite.tgz <paste-url-here>"
    echo "https://github.com/YosysHQ/oss-cad-suite-build/releases"
    exit 1
  fi

  echo "Extracting..."
  tar xzf oss-cad-suite.tgz
  rm oss-cad-suite.tgz
else
  echo "OSS CAD Suite already present at $EDA_DIR/oss-cad-suite, skipping download."
fi

# ---- 3. Pull the Nangate45 open cell library (liberty + LEF) ----
NANGATE_DIR="$EDA_DIR/nangate45"
if [ ! -f "$NANGATE_DIR/lib/NangateOpenCellLibrary_typical.lib" ]; then
  echo "Fetching Nangate45 library from OpenROAD-flow-scripts (sparse checkout)..."
  rm -rf "$EDA_DIR/orfs-sparse-tmp"
  git clone --filter=blob:none --no-checkout --depth 1 \
    https://github.com/The-OpenROAD-Project/OpenROAD-flow-scripts.git \
    "$EDA_DIR/orfs-sparse-tmp"
  cd "$EDA_DIR/orfs-sparse-tmp"
  git sparse-checkout set flow/platforms/nangate45
  git checkout
  cd "$EDA_DIR"
  mkdir -p "$NANGATE_DIR/lib" "$NANGATE_DIR/lef"
  cp orfs-sparse-tmp/flow/platforms/nangate45/lib/*.lib "$NANGATE_DIR/lib/" 2>/dev/null || true
  cp orfs-sparse-tmp/flow/platforms/nangate45/lef/*.lef "$NANGATE_DIR/lef/" 2>/dev/null || true
  rm -rf "$EDA_DIR/orfs-sparse-tmp"
else
  echo "Nangate45 library already present, skipping."
fi

echo ""
echo "==================================================================="
echo "Done. Before running synthesis, in every new shell run:"
echo "  source $EDA_DIR/oss-cad-suite/environment"
echo ""
echo "Sanity check afterward:"
echo "  yosys -V"
echo "  iverilog -V"
echo "  verilator --version"
echo "  ls $NANGATE_DIR/lib"
echo "==================================================================="