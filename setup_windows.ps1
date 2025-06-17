#!/usr/bin/env pwsh
# Ultimate Arbitrage System - Windows Setup Script
# ===============================================
# 
# Comprehensive setup script for Windows environment
# Installs all dependencies and configures the system

Param(
    [switch]$InstallRust,
    [switch]$InstallPython,
    [switch]$InstallDependencies,
    [switch]$BuildRust,
    [switch]$RunTests,
    [switch]$RunDemo,
    [switch]$All,
    [switch]$Force
)

# Set error handling
$ErrorActionPreference = "Stop"

# Color functions for better output
function Write-ColorOutput {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Message,
        [Parameter(Mandatory = $false)]
        [ConsoleColor]$ForegroundColor = "White"
    )
    Write-Host $Message -ForegroundColor $ForegroundColor
}

function Write-Header {
    param([string]$Title)
    Write-Host ""
    Write-ColorOutput "=" * 80 -ForegroundColor Cyan
    Write-ColorOutput "🚀 $Title" -ForegroundColor Yellow
    Write-ColorOutput "=" * 80 -ForegroundColor Cyan
    Write-Host ""
}

function Write-Step {
    param([string]$Step)
    Write-ColorOutput "⚡ $Step" -ForegroundColor Green
}

function Write-Success {
    param([string]$Message)
    Write-ColorOutput "✅ $Message" -ForegroundColor Green
}

function Write-Warning {
    param([string]$Message)
    Write-ColorOutput "⚠️ $Message" -ForegroundColor Yellow
}

function Write-Error-Custom {
    param([string]$Message)
    Write-ColorOutput "❌ $Message" -ForegroundColor Red
}

function Test-CommandExists {
    param([string]$Command)
    try {
        Get-Command $Command -ErrorAction Stop | Out-Null
        return $true
    }
    catch {
        return $false
    }
}

function Install-Rust {
    Write-Step "Installing Rust toolchain..."
    
    if (Test-CommandExists "rustc") {
        Write-Success "Rust is already installed"
        rustc --version
        return
    }
    
    try {
        # Download and install Rust
        Write-ColorOutput "Downloading Rust installer..." -ForegroundColor Blue
        $rustupUrl = "https://win.rustup.rs/x86_64"
        $rustupPath = "$env:TEMP\rustup-init.exe"
        
        Invoke-WebRequest -Uri $rustupUrl -OutFile $rustupPath
        
        Write-ColorOutput "Installing Rust (this may take several minutes)..." -ForegroundColor Blue
        & $rustupPath -y --default-toolchain stable
        
        # Refresh environment variables
        $env:PATH = [System.Environment]::GetEnvironmentVariable("Path", "Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path", "User")
        
        Write-Success "Rust installed successfully"
        rustc --version
        cargo --version
    }
    catch {
        Write-Error-Custom "Failed to install Rust: $_"
        throw
    }
}

function Install-VisualStudioBuildTools {
    Write-Step "Checking for Visual Studio Build Tools..."
    
    # Check if Visual Studio Build Tools are installed
    $buildToolsPath = "${env:ProgramFiles(x86)}\Microsoft Visual Studio\Installer\vswhere.exe"
    if (Test-Path $buildToolsPath) {
        $installations = & $buildToolsPath -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
        if ($installations) {
            Write-Success "Visual Studio Build Tools found"
            return
        }
    }
    
    Write-Warning "Visual Studio Build Tools not found. Rust compilation may fail."
    Write-ColorOutput "Please install Visual Studio Build Tools manually:" -ForegroundColor Yellow
    Write-ColorOutput "1. Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/" -ForegroundColor Yellow
    Write-ColorOutput "2. Install with C++ build tools workload" -ForegroundColor Yellow
    Write-ColorOutput "3. Restart this script after installation" -ForegroundColor Yellow
    
    if (-not $Force) {
        $choice = Read-Host "Continue without Build Tools? (y/N)"
        if ($choice -ne "y" -and $choice -ne "Y") {
            throw "Visual Studio Build Tools required for Rust compilation"
        }
    }
}

function Install-Python {
    Write-Step "Checking Python installation..."
    
    if (Test-CommandExists "python") {
        $pythonVersion = python --version 2>&1
        Write-Success "Python is already installed: $pythonVersion"
        return
    }
    
    Write-Warning "Python not found. Please install Python 3.8 or later manually."
    Write-ColorOutput "Download from: https://www.python.org/downloads/" -ForegroundColor Yellow
    
    if (-not $Force) {
        $choice = Read-Host "Continue without Python? (y/N)"
        if ($choice -ne "y" -and $choice -ne "Y") {
            throw "Python is required"
        }
    }
}

function Install-PythonDependencies {
    Write-Step "Installing Python dependencies..."
    
    if (-not (Test-CommandExists "python")) {
        Write-Warning "Python not found, skipping Python dependencies"
        return
    }
    
    try {
        # Upgrade pip
        Write-ColorOutput "Upgrading pip..." -ForegroundColor Blue
        python -m pip install --upgrade pip
        
        # Install core dependencies
        Write-ColorOutput "Installing core dependencies..." -ForegroundColor Blue
        python -m pip install -r requirements.txt
        
        # Install development dependencies
        Write-ColorOutput "Installing development dependencies..." -ForegroundColor Blue
        python -m pip install pytest pytest-asyncio colorama numpy pandas
        
        Write-Success "Python dependencies installed successfully"
    }
    catch {
        Write-Error-Custom "Failed to install Python dependencies: $_"
        throw
    }
}

function Build-RustEngine {
    Write-Step "Building Rust engine..."
    
    if (-not (Test-CommandExists "cargo")) {
        Write-Error-Custom "Cargo not found. Please install Rust first."
        return $false
    }
    
    try {
        Push-Location "rust_engine"
        
        Write-ColorOutput "Checking Rust toolchain..." -ForegroundColor Blue
        cargo --version
        rustc --version
        
        Write-ColorOutput "Building Rust components..." -ForegroundColor Blue
        cargo build --release
        
        Write-ColorOutput "Running Rust tests..." -ForegroundColor Blue
        cargo test
        
        Write-Success "Rust engine built successfully"
        return $true
    }
    catch {
        Write-Error-Custom "Failed to build Rust engine: $_"
        Write-ColorOutput "This may be due to missing Visual Studio Build Tools" -ForegroundColor Yellow
        return $false
    }
    finally {
        Pop-Location
    }
}

function Build-PythonBindings {
    Write-Step "Building Python bindings..."
    
    if (-not (Test-CommandExists "python")) {
        Write-Warning "Python not found, skipping Python bindings"
        return $false
    }
    
    try {
        Push-Location "rust_engine"
        
        # Install maturin for Python bindings
        Write-ColorOutput "Installing maturin..." -ForegroundColor Blue
        python -m pip install maturin
        
        # Build Python bindings
        Write-ColorOutput "Building Python bindings..." -ForegroundColor Blue
        maturin develop --release
        
        Write-Success "Python bindings built successfully"
        return $true
    }
    catch {
        Write-Error-Custom "Failed to build Python bindings: $_"
        return $false
    }
    finally {
        Pop-Location
    }
}

function Run-Tests {
    Write-Step "Running comprehensive tests..."
    
    try {
        if (Test-CommandExists "python") {
            Write-ColorOutput "Running Python tests..." -ForegroundColor Blue
            python -m pytest tests/ -v
            
            Write-ColorOutput "Running integration tests..." -ForegroundColor Blue
            python tests/test_rust_integration.py
        }
        
        if (Test-CommandExists "cargo") {
            Push-Location "rust_engine"
            Write-ColorOutput "Running Rust tests..." -ForegroundColor Blue
            cargo test -- --nocapture
            Pop-Location
        }
        
        Write-Success "All tests completed"
    }
    catch {
        Write-Error-Custom "Tests failed: $_"
        throw
    }
}

function Run-Demo {
    Write-Step "Running system demo..."
    
    if (-not (Test-CommandExists "python")) {
        Write-Error-Custom "Python not found. Cannot run demo."
        return
    }
    
    try {
        Write-ColorOutput "Starting Ultimate Arbitrage System Demo..." -ForegroundColor Blue
        python demo/ultimate_system_demo.py
        
        Write-Success "Demo completed successfully"
    }
    catch {
        Write-Error-Custom "Demo failed: $_"
        throw
    }
}

function Show-SystemInfo {
    Write-Header "System Information"
    
    Write-ColorOutput "Operating System: $([System.Environment]::OSVersion.VersionString)" -ForegroundColor Blue
    Write-ColorOutput "PowerShell Version: $($PSVersionTable.PSVersion)" -ForegroundColor Blue
    Write-ColorOutput "Architecture: $([System.Environment]::Is64BitOperatingSystem)" -ForegroundColor Blue
    
    if (Test-CommandExists "python") {
        $pythonVersion = python --version 2>&1
        Write-ColorOutput "Python: $pythonVersion" -ForegroundColor Green
    } else {
        Write-ColorOutput "Python: Not installed" -ForegroundColor Red
    }
    
    if (Test-CommandExists "rustc") {
        $rustVersion = rustc --version
        Write-ColorOutput "Rust: $rustVersion" -ForegroundColor Green
    } else {
        Write-ColorOutput "Rust: Not installed" -ForegroundColor Red
    }
    
    if (Test-CommandExists "cargo") {
        $cargoVersion = cargo --version
        Write-ColorOutput "Cargo: $cargoVersion" -ForegroundColor Green
    } else {
        Write-ColorOutput "Cargo: Not installed" -ForegroundColor Red
    }
}

function Show-Usage {
    Write-Header "Ultimate Arbitrage System Setup"
    
    Write-ColorOutput "Usage: .\setup_windows.ps1 [options]" -ForegroundColor Yellow
    Write-Host ""
    Write-ColorOutput "Options:" -ForegroundColor Cyan
    Write-ColorOutput "  -InstallRust        Install Rust toolchain" -ForegroundColor White
    Write-ColorOutput "  -InstallPython      Check/prompt for Python installation" -ForegroundColor White
    Write-ColorOutput "  -InstallDependencies Install Python dependencies" -ForegroundColor White
    Write-ColorOutput "  -BuildRust          Build Rust engine" -ForegroundColor White
    Write-ColorOutput "  -RunTests           Run all tests" -ForegroundColor White
    Write-ColorOutput "  -RunDemo            Run system demonstration" -ForegroundColor White
    Write-ColorOutput "  -All                Perform complete setup" -ForegroundColor White
    Write-ColorOutput "  -Force              Skip interactive prompts" -ForegroundColor White
    Write-Host ""
    Write-ColorOutput "Examples:" -ForegroundColor Cyan
    Write-ColorOutput "  .\setup_windows.ps1 -All                 # Complete setup" -ForegroundColor Green
    Write-ColorOutput "  .\setup_windows.ps1 -InstallRust -BuildRust  # Rust only" -ForegroundColor Green
    Write-ColorOutput "  .\setup_windows.ps1 -RunDemo             # Run demonstration" -ForegroundColor Green
}

# Main execution
try {
    Write-Header "Ultimate Arbitrage System - Windows Setup"
    
    # Show system information
    Show-SystemInfo
    
    # Handle no parameters
    if (-not ($InstallRust -or $InstallPython -or $InstallDependencies -or $BuildRust -or $RunTests -or $RunDemo -or $All)) {
        Show-Usage
        exit 0
    }
    
    # Perform requested actions
    if ($All -or $InstallRust) {
        Install-VisualStudioBuildTools
        Install-Rust
    }
    
    if ($All -or $InstallPython) {
        Install-Python
    }
    
    if ($All -or $InstallDependencies) {
        Install-PythonDependencies
    }
    
    if ($All -or $BuildRust) {
        $rustBuildSuccess = Build-RustEngine
        if ($rustBuildSuccess) {
            Build-PythonBindings
        }
    }
    
    if ($RunTests) {
        Run-Tests
    }
    
    if ($RunDemo) {
        Run-Demo
    }
    
    Write-Header "Setup Completed Successfully"
    Write-Success "Ultimate Arbitrage System is ready!"
    
    if ($All) {
        Write-Host ""
        Write-ColorOutput "Next steps:" -ForegroundColor Cyan
        Write-ColorOutput "1. Run tests: .\setup_windows.ps1 -RunTests" -ForegroundColor Yellow
        Write-ColorOutput "2. Run demo: .\setup_windows.ps1 -RunDemo" -ForegroundColor Yellow
        Write-ColorOutput "3. Explore the codebase and documentation" -ForegroundColor Yellow
    }
}
catch {
    Write-Error-Custom "Setup failed: $_"
    Write-Host ""
    Write-ColorOutput "Troubleshooting:" -ForegroundColor Yellow
    Write-ColorOutput "1. Ensure you're running PowerShell as Administrator" -ForegroundColor White
    Write-ColorOutput "2. Check internet connection for downloads" -ForegroundColor White
    Write-ColorOutput "3. Install Visual Studio Build Tools manually if needed" -ForegroundColor White
    Write-ColorOutput "4. Run with -Force to skip interactive prompts" -ForegroundColor White
    exit 1
}

