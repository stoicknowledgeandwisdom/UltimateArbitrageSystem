#!/bin/bash

echo "Installing Wallet Dashboard dependencies..."
echo

# Check if Node.js is installed
if ! command -v node &> /dev/null; then
    echo "Node.js is not installed. Please install Node.js first."
    echo "Download from: https://nodejs.org/"
    exit 1
fi

# Install dependencies
echo "Installing npm dependencies..."
npm install

if [ $? -ne 0 ]; then
    echo "Failed to install dependencies."
    exit 1
fi

# Create environment file
if [ ! -f .env.local ]; then
    echo "Creating .env.local file..."
    cat > .env.local << EOF
NEXT_PUBLIC_WS_URL=ws://localhost:8080/ws
NEXT_PUBLIC_API_URL=http://localhost:8080/api
EOF
    echo "Environment file created."
fi

echo
echo "Installation complete!"
echo
echo "To start the development server, run:"
echo "  npm run dev"
echo
echo "To build for production, run:"
echo "  npm run build"
echo
echo "For Tauri desktop app, run:"
echo "  npm run tauri:dev"
echo

