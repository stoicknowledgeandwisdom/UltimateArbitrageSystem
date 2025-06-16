/**
 * Format currency values with proper locale and symbol
 */
export function formatCurrency(amount: number, currency: string): string {
  try {
    // Handle crypto currencies
    if (['BTC', 'ETH', 'USDT', 'USDC', 'BNB', 'ADA', 'SOL', 'DOT', 'LINK'].includes(currency)) {
      const precision = currency === 'BTC' ? 8 : currency === 'ETH' ? 6 : 2
      return `${amount.toLocaleString('en-US', {
        minimumFractionDigits: precision,
        maximumFractionDigits: precision
      })} ${currency}`
    }
    
    // Handle fiat currencies
    return new Intl.NumberFormat('en-US', {
      style: 'currency',
      currency: currency === 'USDT' || currency === 'USDC' ? 'USD' : currency,
      minimumFractionDigits: 2,
      maximumFractionDigits: 2
    }).format(amount)
  } catch (error) {
    // Fallback formatting
    return `${amount.toLocaleString('en-US')} ${currency}`
  }
}

/**
 * Format wallet addresses with ellipsis
 */
export function formatAddress(address: string, startChars = 6, endChars = 4): string {
  if (!address) return ''
  if (address.length <= startChars + endChars) return address
  
  return `${address.slice(0, startChars)}...${address.slice(-endChars)}`
}

/**
 * Format percentage values
 */
export function formatPercentage(value: number, precision = 2): string {
  const sign = value >= 0 ? '+' : ''
  return `${sign}${value.toFixed(precision)}%`
}

/**
 * Format large numbers with K, M, B suffixes
 */
export function formatCompactNumber(value: number): string {
  const absValue = Math.abs(value)
  
  if (absValue >= 1000000000) {
    return `${(value / 1000000000).toFixed(1)}B`
  } else if (absValue >= 1000000) {
    return `${(value / 1000000).toFixed(1)}M`
  } else if (absValue >= 1000) {
    return `${(value / 1000).toFixed(1)}K`
  }
  
  return value.toFixed(2)
}

/**
 * Format time duration
 */
export function formatDuration(seconds: number): string {
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const secs = seconds % 60
  
  if (hours > 0) {
    return `${hours}h ${minutes}m`
  } else if (minutes > 0) {
    return `${minutes}m ${secs}s`
  } else {
    return `${secs}s`
  }
}

/**
 * Format date relative to now
 */
export function formatRelativeTime(date: Date): string {
  const now = new Date()
  const diffMs = now.getTime() - date.getTime()
  const diffSeconds = Math.floor(diffMs / 1000)
  const diffMinutes = Math.floor(diffSeconds / 60)
  const diffHours = Math.floor(diffMinutes / 60)
  const diffDays = Math.floor(diffHours / 24)
  
  if (diffSeconds < 60) {
    return 'just now'
  } else if (diffMinutes < 60) {
    return `${diffMinutes}m ago`
  } else if (diffHours < 24) {
    return `${diffHours}h ago`
  } else if (diffDays < 7) {
    return `${diffDays}d ago`
  } else {
    return date.toLocaleDateString()
  }
}

/**
 * Format PnL with appropriate colors
 */
export function formatPnL(value: number, currency = 'USD'): {
  formatted: string
  className: string
  isPositive: boolean
} {
  const isPositive = value >= 0
  const formatted = formatCurrency(Math.abs(value), currency)
  const sign = isPositive ? '+' : '-'
  
  return {
    formatted: `${sign}${formatted}`,
    className: isPositive ? 'pnl-positive' : 'pnl-negative',
    isPositive
  }
}

/**
 * Format trading volume
 */
export function formatVolume(volume: number, currency = 'USD'): string {
  if (volume >= 1000000) {
    return `${(volume / 1000000).toFixed(1)}M ${currency}`
  } else if (volume >= 1000) {
    return `${(volume / 1000).toFixed(1)}K ${currency}`
  }
  return formatCurrency(volume, currency)
}

/**
 * Format file size
 */
export function formatFileSize(bytes: number): string {
  if (bytes === 0) return '0 Bytes'
  
  const k = 1024
  const sizes = ['Bytes', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  
  return `${parseFloat((bytes / Math.pow(k, i)).toFixed(2))} ${sizes[i]}`
}

/**
 * Format transaction hash
 */
export function formatTxHash(hash: string): string {
  return formatAddress(hash, 8, 6)
}

/**
 * Format gas price in Gwei
 */
export function formatGasPrice(wei: number): string {
  const gwei = wei / 1000000000
  return `${gwei.toFixed(2)} Gwei`
}

/**
 * Format risk score
 */
export function formatRiskScore(score: number): {
  level: 'low' | 'medium' | 'high'
  color: string
  label: string
} {
  if (score <= 0.3) {
    return {
      level: 'low',
      color: 'text-success-600',
      label: 'Low Risk'
    }
  } else if (score <= 0.7) {
    return {
      level: 'medium',
      color: 'text-warning-600',
      label: 'Medium Risk'
    }
  } else {
    return {
      level: 'high',
      color: 'text-danger-600',
      label: 'High Risk'
    }
  }
}

