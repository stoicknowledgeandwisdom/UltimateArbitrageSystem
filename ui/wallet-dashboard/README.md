# Wallet & Account Management Dashboard

Advanced crypto trading and portfolio management dashboard built with Next.js, TypeScript, and Tailwind CSS.

## Features

### 🏦 Multi-Wallet Management
- Exchange API keys (Binance, Coinbase, etc.)
- On-chain wallet addresses (MetaMask, WalletConnect)
- Cold storage integration (Ledger, Trezor)
- Real-time balance updates

### 📊 Live Portfolio Analytics
- Real-time P&L tracking
- Risk heat-map visualization
- Strategy performance monitoring
- Position management

### 🔄 Transfer Wizard
- On-chain fee estimation
- Bridge route optimization (RFQ)
- Cross-chain transfer support
- Transaction history

### 🔐 Enterprise Security
- Client-side encryption (WebCrypto API)
- Hardware wallet integration
- QR code air-gapped signing
- Role-based access control

### 📋 Compliance & Audit
- Complete audit trail
- Export in multiple formats (CSV, Parquet, XBRL)
- Regulatory compliance tools
- Transaction reporting

### 🧪 Built-in Simulator
- Strategy backtesting
- Parameter optimization
- YAML configuration export
- Risk scenario modeling

## Tech Stack

- **Frontend**: Next.js 14, TypeScript, Tailwind CSS
- **State Management**: Zustand
- **Animations**: Framer Motion
- **Charts**: Chart.js, Recharts
- **PWA**: Next-PWA
- **Desktop**: Tauri (optional)
- **Security**: WebCrypto API, Ledger SDK

## Quick Start

```bash
# Install dependencies
npm install

# Start development server
npm run dev

# Build for production
npm run build

# Start production server
npm start
```

## Tauri Desktop App

```bash
# Install Tauri CLI
cargo install tauri-cli

# Run desktop app in development
npm run tauri:dev

# Build desktop app
npm run tauri:build
```

## Environment Variables

Create a `.env.local` file:

```env
NEXT_PUBLIC_WS_URL=ws://localhost:8080/ws
NEXT_PUBLIC_API_URL=http://localhost:8080/api
```

## Role-Based Access

- **Trader**: Wallet management, trading, transfers
- **Quant**: Full access + simulator and strategy tools
- **Compliance Officer**: Audit logs, reporting, compliance
- **Admin**: Full system access

## Security Features

- All API keys encrypted client-side before storage
- Hardware wallet support for signing
- QR code generation for air-gapped transactions
- Session management and timeout
- IP whitelisting and 2FA support

## License

MIT License - see LICENSE file for details.

