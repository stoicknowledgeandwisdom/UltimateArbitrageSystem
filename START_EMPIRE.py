#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
🚀 ULTIMATE ENHANCED ARBITRAGE EMPIRE - QUICK START 🚀
=====================================================

Simple one-click launcher for the complete arbitrage empire.
Just run this script and everything will be set up automatically!

Zero-Investment Mindset: Transcending All Boundaries
"""

import os
import sys
import subprocess
import platform
from pathlib import Path

def print_banner():
    """Display the startup banner"""
    banner = """
╔══════════════════════════════════════════════════════════════════════════════╗
║                                                                              ║
║    🚀 ULTIMATE ENHANCED ARBITRAGE EMPIRE - QUICK START 🚀                   ║
║                                                                              ║
║           💎 Zero-Investment Mindset Empire Launcher 💎                      ║
║                                                                              ║
║  🔥 Initializing maximum income generation systems...                        ║
║  ⚛️ Loading quantum-enhanced optimization algorithms...                      ║
║  🧠 Activating AI-powered intelligence engines...                           ║
║  💰 Preparing ultra-high-frequency profit extraction...                     ║
║                                                                              ║
║               🏆 TRANSCENDING ALL LIMITATIONS! 🏆                           ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
    """
    print(banner)

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Error: Python 3.8+ is required")
        print(f"   Current version: {version.major}.{version.minor}.{version.micro}")
        print("   Please upgrade Python and try again.")
        return False
    
    print(f"✅ Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies"""
    print("📦 Installing required dependencies...")
    
    # Core dependencies
    dependencies = [
        "fastapi",
        "uvicorn[standard]",
        "websockets",
        "numpy",
        "pandas",
        "aiofiles",
        "httpx",
        "psutil",
        "uvloop",
    ]
    
    # Optional ML dependencies
    ml_dependencies = [
        "scikit-learn",
        "torch",
        "transformers",
    ]
    
    # Install core dependencies
    for dep in dependencies:
        try:
            print(f"  Installing {dep}...")
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            print(f"  ⚠️ Warning: Failed to install {dep}")
    
    # Try to install ML dependencies (optional)
    for dep in ml_dependencies:
        try:
            subprocess.run([sys.executable, "-m", "pip", "install", dep], 
                         check=True, capture_output=True)
        except subprocess.CalledProcessError:
            pass  # ML dependencies are optional
    
    print("✅ Dependencies installation complete")

def create_startup_files():
    """Create any missing startup files"""
    base_dir = Path(__file__).parent
    
    # Create run script for Windows
    if platform.system() == "Windows":
        run_script = base_dir / "RUN_EMPIRE.bat"
        with open(run_script, 'w') as f:
            f.write(f"""@echo off
echo 🚀 Starting Ultimate Enhanced Arbitrage Empire...
cd /d "{base_dir}"
python launch_ultimate_enhanced_empire.py
pause
""")
        print(f"✅ Created Windows launcher: {run_script}")
    
    # Create run script for Unix/Linux/Mac
    else:
        run_script = base_dir / "run_empire.sh"
        with open(run_script, 'w') as f:
            f.write(f"""#!/bin/bash
echo "🚀 Starting Ultimate Enhanced Arbitrage Empire..."
cd "{base_dir}"
python3 launch_ultimate_enhanced_empire.py
""")
        
        # Make script executable
        os.chmod(run_script, 0o755)
        print(f"✅ Created Unix launcher: {run_script}")

def main():
    """Main startup function"""
    print_banner()
    
    print("🔍 Performing system checks...")
    
    # Check Python version
    if not check_python_version():
        input("Press Enter to exit...")
        return
    
    # Install dependencies
    try:
        install_dependencies()
    except Exception as e:
        print(f"⚠️ Warning: Some dependencies may not have installed correctly: {e}")
    
    # Create startup files
    create_startup_files()
    
    print("\n🎯 System ready! Starting Ultimate Enhanced Arbitrage Empire...")
    print("=" * 80)
    
    # Launch the main empire
    try:
        base_dir = Path(__file__).parent
        empire_launcher = base_dir / "launch_ultimate_enhanced_empire.py"
        
        if empire_launcher.exists():
            subprocess.run([sys.executable, str(empire_launcher)])
        else:
            print("❌ Error: Empire launcher not found!")
            print("   Please ensure launch_ultimate_enhanced_empire.py exists in the same directory.")
    
    except KeyboardInterrupt:
        print("\n👑 Empire startup cancelled by user")
    except Exception as e:
        print(f"\n❌ Error starting empire: {e}")
        print("🔧 Troubleshooting tips:")
        print("   1. Ensure all files are in the correct directory")
        print("   2. Check that Python 3.8+ is installed")
        print("   3. Try running: pip install fastapi uvicorn websockets")
    
    finally:
        print("\n🚀 Ultimate Enhanced Arbitrage Empire - Session Complete")
        if platform.system() == "Windows":
            input("Press Enter to exit...")

if __name__ == "__main__":
    main()

