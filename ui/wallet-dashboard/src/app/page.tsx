'use client'

import { useState, useEffect } from 'react'
import { DashboardLayout } from '@/components/layout/DashboardLayout'
import { WalletOverview } from '@/components/wallet/WalletOverview'
import { PortfolioPnL } from '@/components/portfolio/PortfolioPnL'
import { RiskHeatmap } from '@/components/risk/RiskHeatmap'
import { StrategyHealth } from '@/components/strategy/StrategyHealth'
import { TransferWizard } from '@/components/transfer/TransferWizard'
import { AuditLog } from '@/components/compliance/AuditLog'
import { HardwareWalletPanel } from '@/components/security/HardwareWalletPanel'
import { RoleSelector } from '@/components/auth/RoleSelector'
import { SimulatorPanel } from '@/components/simulator/SimulatorPanel'
import { useWebSocket } from '@/hooks/useWebSocket'
import { useStore } from '@/stores/useStore'
import { Wallet, Position, Strategy } from '@/types'

export default function Dashboard() {
  const [activeView, setActiveView] = useState('overview')
  const { connected: wsConnected } = useWebSocket('ws://localhost:8080/ws')
  const { wallets, positions, strategies, user } = useStore()

  // Mock data for demonstration - replace with actual API calls
  useEffect(() => {
    // Initialize mock data
    const mockWallets: Wallet[] = [
      {
        id: '1',
        name: 'Binance Main',
        type: 'exchange',
        exchangeId: 'binance',
        balance: 125000,
        currency: 'USDT',
        status: 'active',
        lastUpdated: new Date(),
      },
      {
        id: '2',
        name: 'MetaMask Wallet',
        type: 'onchain',
        address: '0x742d35Cc6639C0532bD5e9d24c0da3f74c4aa836',
        balance: 50000,
        currency: 'ETH',
        status: 'active',
        lastUpdated: new Date(),
      },
      {
        id: '3',
        name: 'Ledger Cold Storage',
        type: 'cold_storage',
        address: '0x1a2b3c4d5e6f7g8h9i0j',
        balance: 250000,
        currency: 'BTC',
        status: 'active',
        lastUpdated: new Date(),
        hardwareWalletConnected: true,
      },
    ]

    // Set mock data in store
    useStore.setState({ wallets: mockWallets })
  }, [])

  const renderContent = () => {
    switch (activeView) {
      case 'overview':
        return (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-2 xl:grid-cols-3 gap-6">
              <WalletOverview wallets={wallets} />
              <PortfolioPnL positions={positions} />
              <StrategyHealth strategies={strategies} />
            </div>
            <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
              <RiskHeatmap />
              <div className="space-y-4">
                <HardwareWalletPanel />
                <div className="flex items-center justify-between">
                  <h3 className="text-lg font-semibold">WebSocket Status</h3>
                  <div className={`flex items-center space-x-2 ${
                    wsConnected ? 'text-success-600' : 'text-danger-600'
                  }`}>
                    <div className={`w-2 h-2 rounded-full ${
                      wsConnected ? 'bg-success-500' : 'bg-danger-500'
                    }`} />
                    <span className="text-sm">
                      {wsConnected ? 'Connected' : 'Disconnected'}
                    </span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )
      case 'wallets':
        return <WalletOverview wallets={wallets} detailed />
      case 'transfers':
        return <TransferWizard />
      case 'compliance':
        return <AuditLog />
      case 'simulator':
        return <SimulatorPanel />
      default:
        return <div>View not found</div>
    }
  }

  return (
    <DashboardLayout
      activeView={activeView}
      onViewChange={setActiveView}
      user={user}
    >
      <div className="flex-1 overflow-auto">
        <div className="p-6">
          <div className="mb-6">
            <RoleSelector currentRole={user?.role} />
          </div>
          {renderContent()}
        </div>
      </div>
    </DashboardLayout>
  )
}

