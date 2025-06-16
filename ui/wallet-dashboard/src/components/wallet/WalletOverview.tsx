'use client'

import { useState } from 'react'
import { motion } from 'framer-motion'
import {
  WalletIcon,
  PlusIcon,
  EyeIcon,
  EyeSlashIcon,
  ArrowUpIcon,
  ArrowDownIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline'
import { Wallet } from '@/types'
import { formatCurrency, formatAddress } from '@/utils/formatters'

interface WalletOverviewProps {
  wallets: Wallet[]
  detailed?: boolean
}

export function WalletOverview({ wallets, detailed = false }: WalletOverviewProps) {
  const [showBalances, setShowBalances] = useState(true)
  const [selectedWallet, setSelectedWallet] = useState<string | null>(null)

  const totalValue = wallets.reduce((sum, wallet) => sum + wallet.balance, 0)
  
  const getWalletIcon = (type: Wallet['type']) => {
    switch (type) {
      case 'exchange':
        return '🏦'
      case 'onchain':
        return '⛓️'
      case 'cold_storage':
        return '🔒'
      default:
        return '💼'
    }
  }

  const getStatusColor = (status: Wallet['status']) => {
    switch (status) {
      case 'active':
        return 'bg-success-100 text-success-800 border-success-200'
      case 'inactive':
        return 'bg-gray-100 text-gray-800 border-gray-200'
      case 'error':
        return 'bg-danger-100 text-danger-800 border-danger-200'
      default:
        return 'bg-gray-100 text-gray-800 border-gray-200'
    }
  }

  const WalletCard = ({ wallet }: { wallet: Wallet }) => (
    <motion.div
      layout
      whileHover={{ scale: 1.02 }}
      className={`card p-6 cursor-pointer transition-all duration-200 ${
        selectedWallet === wallet.id ? 'ring-2 ring-primary-500' : ''
      }`}
      onClick={() => setSelectedWallet(selectedWallet === wallet.id ? null : wallet.id)}
    >
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="text-2xl">{getWalletIcon(wallet.type)}</div>
          <div>
            <h3 className="font-semibold text-gray-900 dark:text-white">
              {wallet.name}
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400 capitalize">
              {wallet.type.replace('_', ' ')}
            </p>
          </div>
        </div>
        
        <div className="flex items-center space-x-2">
          <span className={`px-2 py-1 rounded-full text-xs font-medium border ${
            getStatusColor(wallet.status)
          }`}>
            {wallet.status}
          </span>
          {wallet.hardwareWalletConnected && (
            <div className="w-2 h-2 bg-success-500 rounded-full" title="Hardware wallet connected" />
          )}
        </div>
      </div>

      <div className="space-y-3">
        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-500 dark:text-gray-400">Balance</span>
          <div className="text-right">
            <div className="font-semibold text-gray-900 dark:text-white">
              {showBalances ? (
                formatCurrency(wallet.balance, wallet.currency)
              ) : (
                '••••••'
              )}
            </div>
            <div className="text-xs text-gray-500 dark:text-gray-400">
              {wallet.currency}
            </div>
          </div>
        </div>

        {wallet.address && (
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-500 dark:text-gray-400">Address</span>
            <div className="font-mono text-sm text-gray-600 dark:text-gray-300">
              {formatAddress(wallet.address)}
            </div>
          </div>
        )}

        {wallet.exchangeId && (
          <div className="flex items-center justify-between">
            <span className="text-sm text-gray-500 dark:text-gray-400">Exchange</span>
            <div className="text-sm text-gray-600 dark:text-gray-300 capitalize">
              {wallet.exchangeId}
            </div>
          </div>
        )}

        <div className="flex items-center justify-between">
          <span className="text-sm text-gray-500 dark:text-gray-400">Last Updated</span>
          <div className="text-sm text-gray-600 dark:text-gray-300">
            {wallet.lastUpdated.toLocaleTimeString()}
          </div>
        </div>
      </div>

      {selectedWallet === wallet.id && detailed && (
        <motion.div
          initial={{ opacity: 0, height: 0 }}
          animate={{ opacity: 1, height: 'auto' }}
          exit={{ opacity: 0, height: 0 }}
          className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700"
        >
          <div className="flex space-x-2">
            <button className="btn-primary text-sm py-1 px-3 flex items-center space-x-1">
              <ArrowUpIcon className="w-4 h-4" />
              <span>Send</span>
            </button>
            <button className="btn-secondary text-sm py-1 px-3 flex items-center space-x-1">
              <ArrowDownIcon className="w-4 h-4" />
              <span>Receive</span>
            </button>
            <button className="btn-secondary text-sm py-1 px-3 flex items-center space-x-1">
              <Cog6ToothIcon className="w-4 h-4" />
              <span>Settings</span>
            </button>
          </div>
        </motion.div>
      )}
    </motion.div>
  )

  if (!detailed) {
    return (
      <div className="card p-6">
        <div className="flex items-center justify-between mb-6">
          <div className="flex items-center space-x-3">
            <WalletIcon className="w-6 h-6 text-primary-600" />
            <h2 className="text-xl font-bold text-gray-900 dark:text-white">
              Wallets
            </h2>
          </div>
          
          <div className="flex items-center space-x-2">
            <button
              onClick={() => setShowBalances(!showBalances)}
              className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
            >
              {showBalances ? (
                <EyeSlashIcon className="w-5 h-5 text-gray-500" />
              ) : (
                <EyeIcon className="w-5 h-5 text-gray-500" />
              )}
            </button>
            <button className="btn-primary flex items-center space-x-1">
              <PlusIcon className="w-4 h-4" />
              <span>Add Wallet</span>
            </button>
          </div>
        </div>

        <div className="space-y-4">
          <div className="bg-gradient-to-r from-primary-50 to-primary-100 dark:from-primary-900/20 dark:to-primary-800/20 rounded-lg p-4">
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm text-primary-600 dark:text-primary-400 font-medium">
                  Total Portfolio Value
                </p>
                <p className="text-2xl font-bold text-primary-900 dark:text-primary-100">
                  {showBalances ? formatCurrency(totalValue, 'USD') : '••••••••'}
                </p>
              </div>
              <div className="text-3xl">
                💰
              </div>
            </div>
          </div>

          <div className="grid grid-cols-1 gap-3">
            {wallets.slice(0, 3).map((wallet) => (
              <div key={wallet.id} className="flex items-center justify-between p-3 rounded-lg bg-gray-50 dark:bg-gray-800">
                <div className="flex items-center space-x-3">
                  <div className="text-lg">{getWalletIcon(wallet.type)}</div>
                  <div>
                    <p className="font-medium text-gray-900 dark:text-white">
                      {wallet.name}
                    </p>
                    <p className="text-sm text-gray-500 dark:text-gray-400">
                      {wallet.currency}
                    </p>
                  </div>
                </div>
                <div className="text-right">
                  <p className="font-semibold text-gray-900 dark:text-white">
                    {showBalances ? formatCurrency(wallet.balance, wallet.currency) : '••••••'}
                  </p>
                  <div className={`w-2 h-2 rounded-full ml-auto ${
                    wallet.status === 'active' ? 'bg-success-500' : 
                    wallet.status === 'error' ? 'bg-danger-500' : 'bg-gray-400'
                  }`} />
                </div>
              </div>
            ))}
          </div>

          {wallets.length > 3 && (
            <div className="text-center">
              <button className="text-primary-600 hover:text-primary-700 text-sm font-medium">
                View all {wallets.length} wallets →
              </button>
            </div>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900 dark:text-white">
            Wallets & Accounts
          </h1>
          <p className="text-gray-500 dark:text-gray-400 mt-1">
            Manage your exchange accounts, on-chain wallets, and cold storage
          </p>
        </div>
        
        <div className="flex items-center space-x-3">
          <button
            onClick={() => setShowBalances(!showBalances)}
            className="btn-secondary flex items-center space-x-2"
          >
            {showBalances ? (
              <EyeSlashIcon className="w-4 h-4" />
            ) : (
              <EyeIcon className="w-4 h-4" />
            )}
            <span>{showBalances ? 'Hide' : 'Show'} Balances</span>
          </button>
          <button className="btn-primary flex items-center space-x-2">
            <PlusIcon className="w-4 h-4" />
            <span>Add Wallet</span>
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
        {wallets.map((wallet) => (
          <WalletCard key={wallet.id} wallet={wallet} />
        ))}
      </div>
    </div>
  )
}

