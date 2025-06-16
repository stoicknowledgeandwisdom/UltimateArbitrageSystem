# Deployment Guide

## Environment Setup

### Prerequisites
- Node.js 18+ 
- npm or yarn
- For desktop app: Rust toolchain + Tauri CLI

### Installation

**Windows:**
```cmd
install.bat
```

**Linux/macOS:**
```bash
chmod +x install.sh
./install.sh
```

**Manual:**
```bash
npm install
cp .env.example .env.local
# Edit .env.local with your configuration
```

## Web Application

### Development
```bash
npm run dev
```
Open http://localhost:3000

### Production Build
```bash
npm run build
npm start
```

### Static Export (for CDN)
```bash
NEXT_EXPORT=true npm run build
# Files will be in ./out directory
```

## Desktop Application (Tauri)

### Prerequisites
```bash
# Install Rust
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Install Tauri CLI
cargo install tauri-cli
```

### Development
```bash
npm run tauri:dev
```

### Build Desktop App
```bash
npm run tauri:build
```

Built applications will be in `src-tauri/target/release/bundle/`

## PWA (Progressive Web App)

The application automatically builds as a PWA with:
- Service worker for offline functionality
- App manifest for installation
- Push notification support

### Installation on Mobile
1. Visit the website on mobile browser
2. Tap "Add to Home Screen" when prompted
3. App will install like a native app

## Security Configuration

### Environment Variables

**Required:**
```env
NEXT_PUBLIC_WS_URL=wss://your-websocket-server.com/ws
NEXT_PUBLIC_API_URL=https://your-api-server.com/api
```

**Optional:**
```env
NEXT_PUBLIC_SENTRY_DSN=your-sentry-dsn
NEXT_PUBLIC_AMPLITUDE_API_KEY=your-amplitude-key
NEXT_PUBLIC_ENVIRONMENT=production
```

### Hardware Wallet Configuration

For Ledger integration:
```env
NEXT_PUBLIC_LEDGER_TRANSPORT=webusb
NEXT_PUBLIC_LEDGER_DEBUG=false
```

### WebSocket Security

For production, ensure WebSocket connections use WSS:
```env
NEXT_PUBLIC_WS_URL=wss://secure-websocket.yourdomain.com/ws
```

## Docker Deployment

### Dockerfile
```dockerfile
FROM node:18-alpine AS builder
WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production
COPY . .
RUN npm run build

FROM node:18-alpine AS runner
WORKDIR /app
ENV NODE_ENV production
RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs
COPY --from=builder /app/public ./public
COPY --from=builder --chown=nextjs:nodejs /app/.next/standalone ./
COPY --from=builder --chown=nextjs:nodejs /app/.next/static ./.next/static
USER nextjs
EXPOSE 3000
ENV PORT 3000
CMD ["node", "server.js"]
```

### Docker Compose
```yaml
version: '3.8'
services:
  wallet-dashboard:
    build: .
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_WS_URL=wss://api.yourdomain.com/ws
      - NEXT_PUBLIC_API_URL=https://api.yourdomain.com/api
    restart: unless-stopped
```

## Nginx Configuration

```nginx
server {
    listen 80;
    server_name wallet.yourdomain.com;
    return 301 https://$server_name$request_uri;
}

server {
    listen 443 ssl http2;
    server_name wallet.yourdomain.com;

    ssl_certificate /path/to/cert.pem;
    ssl_certificate_key /path/to/key.pem;

    location / {
        proxy_pass http://localhost:3000;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection 'upgrade';
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
        proxy_cache_bypass $http_upgrade;
    }

    # WebSocket support
    location /ws {
        proxy_pass http://localhost:8080;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## Monitoring & Analytics

### Error Tracking (Sentry)
```bash
npm install @sentry/nextjs
```

### Performance Monitoring
- Built-in Next.js analytics
- Web Vitals tracking
- Custom performance metrics

### Health Checks
```bash
# Health check endpoint
curl https://wallet.yourdomain.com/api/health
```

## Backup & Recovery

### Local Storage Backup
- User preferences automatically synced
- Encrypted keys stored client-side only
- Export/import functionality built-in

### Database Backup (if using backend)
```bash
# Automated backup script
#!/bin/bash
DATE=$(date +%Y%m%d_%H%M%S)
pg_dump wallet_db > backup_$DATE.sql
```

## Troubleshooting

### Common Issues

1. **Build Failures:**
   ```bash
   rm -rf .next node_modules
   npm install
   npm run build
   ```

2. **WebSocket Connection Issues:**
   - Check firewall settings
   - Verify SSL certificates
   - Ensure WebSocket proxy is configured

3. **Hardware Wallet Not Detected:**
   - Enable WebUSB in browser settings
   - Check USB permissions
   - Try different USB port/cable

4. **PWA Installation Issues:**
   - Clear browser cache
   - Check manifest.json validity
   - Verify HTTPS configuration

### Debug Mode
```bash
DEBUG=* npm run dev
```

### Logs
```bash
# Development logs
npm run dev 2>&1 | tee debug.log

# Production logs
PM2_HOME=/opt/pm2 pm2 logs wallet-dashboard
```

