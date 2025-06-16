'use client'

import { motion } from 'framer-motion'
import { CogIcon } from '@heroicons/react/24/outline'
import { Strategy } from '@/types'

interface StrategyHealthProps {
  strategies: Strategy[]
}

export function StrategyHealth({ strategies }: StrategyHealthProps) {
  const mockStrategies = [
    { id: '1', name: 'Arbitrage Bot', status: 'active', performance: 12.5 },
    { id: '2', name: 'Grid Trading', status: 'active', performance: 8.3 },
    { id: '3', name: 'DCA Strategy', status: 'paused', performance: 5.2 },
  ]

  const getStatusColor = (status: string) => {
    switch (status) {
      case 'active': return 'text-success-600'
      case 'paused': return 'text-warning-600'
      case 'error': return 'text-danger-600'
      default: return 'text-gray-600'
    }
  }

  return (
    <div className="card p-6">
      <div className="flex items-center space-x-3 mb-6">
        <CogIcon className="w-6 h-6 text-primary-600" />
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">
          Strategy Health
        </h2>
      </div>
      
      <div className="space-y-3">
        {mockStrategies.map((strategy) => (
          <motion.div
            key={strategy.id}
            whileHover={{ scale: 1.02 }}
            className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800"
          >
            <div>
              <div className="font-medium text-gray-900 dark:text-white">
                {strategy.name}
              </div>
              <div className={`text-sm capitalize ${getStatusColor(strategy.status)}`}>
                {strategy.status}
              </div>
            </div>
            <div className="text-right">
              <div className="text-lg font-semibold text-success-600">
                +{strategy.performance}%
              </div>
              <div className="text-xs text-gray-500">30d</div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

