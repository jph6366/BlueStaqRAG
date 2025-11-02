#!/bin/bash

set -e  # Exit on any error

echo "🚀 Starting BlueStaq installation..."

# Check system requirements
echo "🔍 Checking system requirements..."
FREE_RAM=$(free -g | awk '/^Mem:/{print $2}')
if [ $FREE_RAM -lt 8 ]; then
    echo "⚠️  Warning: Less than 8GB RAM detected. Performance may be impacted."
fi

CPU_CORES=$(nproc)
echo "ℹ️  Detected $CPU_CORES CPU cores"

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    wget \
    git \
    curl

# Install Julia if not present
if ! command -v julia &> /dev/null; then
    echo "🔧 Installing Julia..."
    curl -fsSL https://install.juliaup.org | sh
    source ~/.bashrc
fi

# Clone repository if not already in it
if [ ! -d ".git" ]; then
    echo "📥 Cloning BlueStaq repository..."
    git clone https://github.com/yourusername/BlueStaq.git
    cd BlueStaq
fi

# Create necessary directories
echo "📂 Creating project directories..."
mkdir -p model data/context

# Download the Mistral model
echo "🤖 Downloading Mistral model..."
if [ ! -f "model/mistral-7b-instruct-v0.1.Q4_K_M.gguf" ]; then
    wget -q --show-progress \
        https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.1-GGUF/resolve/main/mistral-7b-instruct-v0.1.Q4_K_M.gguf \
        -O model/mistral-7b-instruct-v0.1.Q4_K_M.gguf
fi

# Verify model download
echo "✅ Verifying model..."
if [ ! -f "model/mistral-7b-instruct-v0.1.Q4_K_M.gguf" ]; then
    echo "❌ Model download failed!"
    exit 1
fi

# Setup Julia environment
echo "📚 Setting up Julia environment..."
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run initial setup and tests
echo "🧪 Running setup validation..."
julia --project=. -e '
using Pkg
Pkg.test()
'

echo "🎉 Installation complete!"
echo
echo "To use BlueStaq, run:"
echo "julia --project=. src/LocalLLM.jl --query \"YOUR QUESTION\" \\"
echo "    --model-path \"model/mistral-7b-instruct-v0.1.Q4_K_M.gguf\" \\"
echo "    --context-path \"data/context\""
