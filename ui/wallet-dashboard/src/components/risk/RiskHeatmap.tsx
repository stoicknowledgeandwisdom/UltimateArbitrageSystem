'use client'

import { motion } from 'framer-motion'
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline'

export function RiskHeatmap() {
  const mockRiskData = [
    { asset: 'BTC', risk: 0.3, exposure: 45000 },
    { asset: 'ETH', risk: 0.5, exposure: 28000 },
    { asset: 'USDT', risk: 0.1, exposure: 50000 },
    { asset: 'BNB', risk: 0.7, exposure: 15000 },
  ]

  const getRiskColor = (risk: number) => {
    if (risk <= 0.3) return 'bg-success-500'
    if (risk <= 0.6) return 'bg-warning-500'
    return 'bg-danger-500'
  }

  return (
    <div className="card p-6">
      <div className="flex items-center space-x-3 mb-6">
        <ExclamationTriangleIcon className="w-6 h-6 text-warning-600" />
        <h2 className="text-xl font-bold text-gray-900 dark:text-white">
          Risk Heatmap
        </h2>
      </div>
      
      <div className="space-y-3">
        {mockRiskData.map((item) => (
          <motion.div
            key={item.asset}
            whileHover={{ scale: 1.02 }}
            className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800"
          >
            <div className="flex items-center space-x-3">
              <div className={`w-3 h-3 rounded-full ${getRiskColor(item.risk)}`} />
              <span className="font-medium text-gray-900 dark:text-white">
                {item.asset}
              </span>
            </div>
            <div className="text-right">
              <div className="text-sm font-medium text-gray-900 dark:text-white">
                ${item.exposure.toLocaleString()}
              </div>
              <div className="text-xs text-gray-500">
                Risk: {(item.risk * 100).toFixed(0)}%
              </div>
            </div>
          </motion.div>
        ))}
      </div>
    </div>
  )
}

