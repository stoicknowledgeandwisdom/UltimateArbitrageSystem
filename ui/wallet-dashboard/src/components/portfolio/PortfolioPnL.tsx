'use client'

import { useState, useMemo } from 'react'
import { motion } from 'framer-motion'
import {
  ChartBarIcon,
  ArrowTrendingUpIcon,
  ArrowTrendingDownIcon,
} from '@heroicons/react/24/outline'
import { Position } from '@/types'
import { formatPnL, formatPercentage } from '@/utils/formatters'

interface PortfolioPnLProps {
  positions: Position[]
}

export function PortfolioPnL({ positions }: PortfolioPnLProps) {
  const [timeframe, setTimeframe] = useState<'1h' | '24h' | '7d' | '30d'>('24h')

  const pnlData = useMemo(() => {
    const totalUnrealized = positions.reduce((sum, pos) => sum + pos.unrealizedPnL, 0)
    const totalRealized = positions.reduce((sum, pos) => sum + pos.realizedPnL, 0)
    const totalPnL = totalUnrealized + totalRealized
    
    // Mock percentage change based on timeframe
    const percentageChange = {
      '1h': 0.25,
      '24h': 2.4,
      '7d': 8.7,
      '30d': 15.3
    }[timeframe]

    return {
      total: totalPnL,
      unrealized: totalUnrealized,
      realized: totalRealized,
      percentage: percentageChange,
      isPositive: totalPnL >= 0
    }
  }, [positions, timeframe])

  const timeframeOptions = [
    { value: '1h', label: '1H' },
    { value: '24h', label: '24H' },
    { value: '7d', label: '7D' },
    { value: '30d', label: '30D' },
  ] as const

  return (
    <div className="card p-6">
      <div className="flex items-center justify-between mb-6">
        <div className="flex items-center space-x-3">
          <ChartBarIcon className="w-6 h-6 text-primary-600" />
          <h2 className="text-xl font-bold text-gray-900 dark:text-white">
            Portfolio P&L
          </h2>
        </div>
        
        <div className="flex space-x-1 bg-gray-100 dark:bg-gray-700 rounded-lg p-1">
          {timeframeOptions.map((option) => (
            <button
              key={option.value}
              onClick={() => setTimeframe(option.value)}
              className={`px-3 py-1 rounded-md text-sm font-medium transition-colors ${
                timeframe === option.value
                  ? 'bg-white dark:bg-gray-600 text-gray-900 dark:text-white shadow-sm'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-white'
              }`}
            >
              {option.label}
            </button>
          ))}
        </div>
      </div>

      <div className="space-y-6">
        {/* Total P&L */}
        <div className="text-center">
          <div className="flex items-center justify-center space-x-2 mb-2">
            {pnlData.isPositive ? (
              <ArrowTrendingUpIcon className="w-6 h-6 text-success-500" />
            ) : (
              <ArrowTrendingDownIcon className="w-6 h-6 text-danger-500" />
            )}
            <span className="text-3xl font-bold">
              <span className={pnlData.isPositive ? 'text-success-600' : 'text-danger-600'}>
                {formatPnL(pnlData.total).formatted}
              </span>
            </span>
          </div>
          <div className="flex items-center justify-center space-x-2">
            <span className="text-sm text-gray-500 dark:text-gray-400">
              {formatPercentage(pnlData.percentage)}
            </span>
            <span className="text-sm text-gray-500 dark:text-gray-400">
              • {timeframe}
            </span>
          </div>
        </div>

        {/* Breakdown */}
        <div className="grid grid-cols-2 gap-4">
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4"
          >
            <div className="text-center">
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">
                Unrealized P&L
              </p>
              <p className={`text-lg font-semibold ${
                pnlData.unrealized >= 0 ? 'text-success-600' : 'text-danger-600'
              }`}>
                {formatPnL(pnlData.unrealized).formatted}
              </p>
            </div>
          </motion.div>
          
          <motion.div
            whileHover={{ scale: 1.02 }}
            className="bg-gray-50 dark:bg-gray-800 rounded-lg p-4"
          >
            <div className="text-center">
              <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">
                Realized P&L
              </p>
              <p className={`text-lg font-semibold ${
                pnlData.realized >= 0 ? 'text-success-600' : 'text-danger-600'
              }`}>
                {formatPnL(pnlData.realized).formatted}
              </p>
            </div>
          </motion.div>
        </div>

        {/* Position Summary */}
        {positions.length > 0 && (
          <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-white mb-3">
              Top Positions
            </h3>
            <div className="space-y-2">
              {positions.slice(0, 3).map((position) => (
                <div
                  key={position.id}
                  className="flex items-center justify-between p-2 rounded hover:bg-gray-50 dark:hover:bg-gray-800"
                >
                  <div className="flex items-center space-x-2">
                    <div className={`w-2 h-2 rounded-full ${
                      position.side === 'long' ? 'bg-success-500' : 'bg-danger-500'
                    }`} />
                    <span className="text-sm font-medium text-gray-900 dark:text-white">
                      {position.symbol}
                    </span>
                  </div>
                  <div className="text-right">
                    <div className={`text-sm font-medium ${
                      position.unrealizedPnL >= 0 ? 'text-success-600' : 'text-danger-600'
                    }`}>
                      {formatPnL(position.unrealizedPnL).formatted}
                    </div>
                    <div className={`text-xs ${
                      position.percentage >= 0 ? 'text-success-500' : 'text-danger-500'
                    }`}>
                      {formatPercentage(position.percentage)}
                    </div>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

