const withPWA = require('next-pwa')({
  dest: 'public',
  register: true,
  skipWaiting: true,
  disable: process.env.NODE_ENV === 'development'
})

/** @type {import('next').NextConfig} */
const nextConfig = {
  experimental: {
    appDir: true,
  },
  images: {
    domains: ['localhost'],
  },
  // Tauri expects a static export
  ...(process.env.TAURI_PLATFORM && {
    output: 'export',
    trailingSlash: true,
    images: { unoptimized: true },
  }),
}

module.exports = withPWA(nextConfig)

