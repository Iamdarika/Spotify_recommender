#!/bin/bash
# Quick Start Script for Music Recommendation Engine
# This script helps you build and test the recommender

set -e  # Exit on error

echo "╔════════════════════════════════════════════════╗"
echo "║   Music Recommendation Engine - Quick Start   ║"
echo "╚════════════════════════════════════════════════╝"
echo ""

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check for required tools
echo "🔍 Checking dependencies..."

if ! command_exists nvcc; then
    echo "❌ ERROR: CUDA compiler (nvcc) not found!"
    echo "   Install CUDA Toolkit: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

if ! command_exists g++; then
    echo "❌ ERROR: g++ compiler not found!"
    echo "   Install with: sudo apt-get install g++ (Ubuntu/Debian)"
    exit 1
fi

echo "✓ CUDA compiler found: $(nvcc --version | grep release)"
echo "✓ g++ compiler found: $(g++ --version | head -n1)"
echo ""

# Check for NVIDIA GPU
echo "🎮 Checking for NVIDIA GPU..."
if command_exists nvidia-smi; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -n1
    echo ""
else
    echo "⚠️  WARNING: nvidia-smi not found. GPU may not be available."
    echo ""
fi

# Build the project
echo "🔨 Building the project..."
make clean 2>/dev/null || true
make

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Build successful!"
    echo ""
else
    echo ""
    echo "❌ Build failed! Please check the error messages above."
    exit 1
fi

# Check if dataset exists
echo "📊 Checking for dataset..."
if [ -f "dataset.csv" ]; then
    echo "✓ Found dataset.csv"
    echo ""
    
    # Ask if user wants to preprocess
    read -p "Would you like to preprocess the dataset now? (y/n) " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        echo "🔄 Preprocessing dataset..."
        ./recommender --preprocess dataset.csv
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "✅ Preprocessing complete!"
            echo ""
            
            # Ask for a test recommendation
            read -p "Would you like to test a recommendation? (y/n) " -n 1 -r
            echo ""
            
            if [[ $REPLY =~ ^[Yy]$ ]]; then
                echo "Enter a song name to search for:"
                read -r song_name
                echo ""
                echo "🎵 Getting recommendations for: $song_name"
                ./recommender --song "$song_name" -n 5
            fi
        fi
    fi
else
    echo "⚠️  No dataset.csv found in current directory."
    echo ""
    echo "Please add your dataset and run:"
    echo "  ./recommender --preprocess dataset.csv"
    echo ""
    echo "See DATASET_INFO.md for information on obtaining datasets."
fi

echo ""
echo "═══════════════════════════════════════════════"
echo "Quick Usage Reference:"
echo "═══════════════════════════════════════════════"
echo ""
echo "Preprocess data:"
echo "  ./recommender --preprocess <csv_file>"
echo ""
echo "Get recommendations by song name:"
echo "  ./recommender --song \"Song Name\" -n 10"
echo ""
echo "Get recommendations by track ID:"
echo "  ./recommender --id \"track_id\" -n 10"
echo ""
echo "For more information, see README.md"
echo "═══════════════════════════════════════════════"
