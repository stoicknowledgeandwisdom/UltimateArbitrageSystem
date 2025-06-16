'use client'

import { ShieldCheckIcon } from '@heroicons/react/24/outline'

export function HardwareWalletPanel() {
  return (
    <div className="card p-4">
      <div className="flex items-center space-x-3 mb-4">
        <ShieldCheckIcon className="w-5 h-5 text-primary-600" />
        <h3 className="font-semibold text-gray-900 dark:text-white">
          Hardware Wallet
        </h3>
      </div>
      <div className="text-sm text-gray-600 dark:text-gray-400">
        <p>No hardware wallet connected</p>
        <button className="btn-primary text-xs mt-2 py-1 px-2">
          Connect Ledger
        </button>
      </div>
    </div>
  )
}

