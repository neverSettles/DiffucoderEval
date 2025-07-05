#!/usr/bin/env python3
"""
Setup and Run Script for DiffuCoder Example
===========================================

This script sets up the environment and runs the DiffuCoder example.
"""

import subprocess
import sys
import os


def run_command(command, description):
    """Run a shell command with error handling."""
    print(f"🔧 {description}")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return result
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return None


def check_gpu():
    """Check if CUDA is available."""
    try:
        result = subprocess.run("nvidia-smi", shell=True, check=True, capture_output=True, text=True)
        print("🔥 GPU Status:")
        print(result.stdout)
        return True
    except subprocess.CalledProcessError:
        print("❌ nvidia-smi not found. Make sure NVIDIA drivers are installed.")
        return False


def install_dependencies():
    """Install Python dependencies."""
    print("📦 Installing dependencies...")
    
    # Check if we're in a virtual environment
    if sys.prefix != sys.base_prefix:
        print("✅ Virtual environment detected")
    else:
        print("⚠️  Not in a virtual environment. Consider using one.")
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        result = run_command("pip install -r requirements.txt", "Installing requirements")
        if result is None:
            return False
    else:
        print("❌ requirements.txt not found")
        return False
    
    return True


def run_example():
    """Run the DiffuCoder example."""
    print("🚀 Running DiffuCoder example...")
    
    if not os.path.exists("diffucoder_example.py"):
        print("❌ diffucoder_example.py not found")
        return False
    
    # Run the example
    result = run_command("python diffucoder_example.py", "Running DiffuCoder example")
    return result is not None


def main():
    """Main setup and run function."""
    print("🎯 DiffuCoder Setup and Run Script")
    print("=" * 50)
    
    # Check GPU
    if not check_gpu():
        print("⚠️  GPU check failed, but continuing anyway...")
    
    # Install dependencies
    if not install_dependencies():
        print("❌ Dependency installation failed")
        return
    
    # Run the example
    print("\n" + "=" * 50)
    choice = input("Would you like to run the DiffuCoder example now? (y/n): ").strip().lower()
    
    if choice in ['y', 'yes']:
        run_example()
    else:
        print("👍 Setup complete! You can run the example later with:")
        print("python diffucoder_example.py")


if __name__ == "__main__":
    main() 