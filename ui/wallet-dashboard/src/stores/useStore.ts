import { create } from 'zustand'
import { devtools, persist } from 'zustand/middleware'
import { 
  Wallet, 
  Position, 
  Strategy, 
  Transfer, 
  AuditLog, 
  UserRole, 
  UserPreferences,
  WSMessage,
  ExchangeAPIKey,
  ComplianceReport
} from '@/types'

interface AppState {
  // User state
  user: {
    id: string
    role: UserRole
    preferences: UserPreferences
  } | null
  
  // Trading data
  wallets: Wallet[]
  positions: Position[]
  strategies: Strategy[]
  transfers: Transfer[]
  auditLogs: AuditLog[]
  exchangeAPIKeys: ExchangeAPIKey[]
  complianceReports: ComplianceReport[]
  
  // WebSocket state
  websocket: {
    connected: boolean
    lastMessage?: WSMessage
    reconnectAttempts: number
  }
  
  // UI state
  ui: {
    sidebarOpen: boolean
    activeView: string
    loading: boolean
    error?: string
    darkMode: boolean
  }
  
  // Actions
  setUser: (user: AppState['user']) => void
  addWallet: (wallet: Wallet) => void
  updateWallet: (id: string, updates: Partial<Wallet>) => void
  removeWallet: (id: string) => void
  
  addPosition: (position: Position) => void
  updatePosition: (id: string, updates: Partial<Position>) => void
  removePosition: (id: string) => void
  
  addStrategy: (strategy: Strategy) => void
  updateStrategy: (id: string, updates: Partial<Strategy>) => void
  removeStrategy: (id: string) => void
  
  addTransfer: (transfer: Transfer) => void
  updateTransfer: (id: string, updates: Partial<Transfer>) => void
  
  addAuditLog: (log: AuditLog) => void
  
  addExchangeAPIKey: (key: ExchangeAPIKey) => void
  updateExchangeAPIKey: (id: string, updates: Partial<ExchangeAPIKey>) => void
  removeExchangeAPIKey: (id: string) => void
  
  addComplianceReport: (report: ComplianceReport) => void
  updateComplianceReport: (id: string, updates: Partial<ComplianceReport>) => void
  
  setWebSocketState: (state: Partial<AppState['websocket']>) => void
  setUIState: (state: Partial<AppState['ui']>) => void
  
  // Computed values
  getTotalPortfolioValue: () => number
  getTotalPnL: () => number
  getActiveStrategies: () => Strategy[]
  getWalletsByType: (type: Wallet['type']) => Wallet[]
}

const defaultUser: AppState['user'] = {
  id: 'demo-user',
  role: {
    id: 'trader',
    name: 'trader',
    permissions: [
      { resource: 'wallets', actions: ['read', 'create', 'update'] },
      { resource: 'positions', actions: ['read'] },
      { resource: 'strategies', actions: ['read', 'create', 'update'] },
      { resource: 'transfers', actions: ['read', 'create'] },
    ],
    restrictions: [],
  },
  preferences: {
    theme: 'dark',
    currency: 'USDT',
    notifications: {
      email: true,
      push: true,
      trades: true,
      pnl: true,
      security: true,
    },
    dashboard: {
      widgets: [],
      layout: 'grid',
    },
  },
}

export const useStore = create<AppState>()()
  devtools(
    persist(
      (set, get) => ({
        // Initial state
        user: defaultUser,
        wallets: [],
        positions: [],
        strategies: [],
        transfers: [],
        auditLogs: [],
        exchangeAPIKeys: [],
        complianceReports: [],
        websocket: {
          connected: false,
          reconnectAttempts: 0,
        },
        ui: {
          sidebarOpen: true,
          activeView: 'overview',
          loading: false,
          darkMode: true,
        },
        
        // User actions
        setUser: (user) => set({ user }),
        
        // Wallet actions
        addWallet: (wallet) => set((state) => ({ 
          wallets: [...state.wallets, wallet] 
        })),
        updateWallet: (id, updates) => set((state) => ({
          wallets: state.wallets.map(w => w.id === id ? { ...w, ...updates } : w)
        })),
        removeWallet: (id) => set((state) => ({
          wallets: state.wallets.filter(w => w.id !== id)
        })),
        
        // Position actions
        addPosition: (position) => set((state) => ({ 
          positions: [...state.positions, position] 
        })),
        updatePosition: (id, updates) => set((state) => ({
          positions: state.positions.map(p => p.id === id ? { ...p, ...updates } : p)
        })),
        removePosition: (id) => set((state) => ({
          positions: state.positions.filter(p => p.id !== id)
        })),
        
        // Strategy actions
        addStrategy: (strategy) => set((state) => ({ 
          strategies: [...state.strategies, strategy] 
        })),
        updateStrategy: (id, updates) => set((state) => ({
          strategies: state.strategies.map(s => s.id === id ? { ...s, ...updates } : s)
        })),
        removeStrategy: (id) => set((state) => ({
          strategies: state.strategies.filter(s => s.id !== id)
        })),
        
        // Transfer actions
        addTransfer: (transfer) => set((state) => ({ 
          transfers: [...state.transfers, transfer] 
        })),
        updateTransfer: (id, updates) => set((state) => ({
          transfers: state.transfers.map(t => t.id === id ? { ...t, ...updates } : t)
        })),
        
        // Audit log actions
        addAuditLog: (log) => set((state) => ({ 
          auditLogs: [log, ...state.auditLogs].slice(0, 1000) // Keep last 1000 logs
        })),
        
        // Exchange API key actions
        addExchangeAPIKey: (key) => set((state) => ({ 
          exchangeAPIKeys: [...state.exchangeAPIKeys, key] 
        })),
        updateExchangeAPIKey: (id, updates) => set((state) => ({
          exchangeAPIKeys: state.exchangeAPIKeys.map(k => k.id === id ? { ...k, ...updates } : k)
        })),
        removeExchangeAPIKey: (id) => set((state) => ({
          exchangeAPIKeys: state.exchangeAPIKeys.filter(k => k.id !== id)
        })),
        
        // Compliance report actions
        addComplianceReport: (report) => set((state) => ({ 
          complianceReports: [...state.complianceReports, report] 
        })),
        updateComplianceReport: (id, updates) => set((state) => ({
          complianceReports: state.complianceReports.map(r => r.id === id ? { ...r, ...updates } : r)
        })),
        
        // WebSocket actions
        setWebSocketState: (wsState) => set((state) => ({
          websocket: { ...state.websocket, ...wsState }
        })),
        
        // UI actions
        setUIState: (uiState) => set((state) => ({
          ui: { ...state.ui, ...uiState }
        })),
        
        // Computed values
        getTotalPortfolioValue: () => {
          const { wallets } = get()
          return wallets.reduce((total, wallet) => total + wallet.balance, 0)
        },
        
        getTotalPnL: () => {
          const { positions } = get()
          return positions.reduce((total, position) => total + position.unrealizedPnL, 0)
        },
        
        getActiveStrategies: () => {
          const { strategies } = get()
          return strategies.filter(s => s.status === 'active')
        },
        
        getWalletsByType: (type) => {
          const { wallets } = get()
          return wallets.filter(w => w.type === type)
        },
      }),
      {
        name: 'wallet-dashboard-storage',
        partialize: (state) => ({
          user: state.user,
          wallets: state.wallets,
          exchangeAPIKeys: state.exchangeAPIKeys,
          ui: {
            sidebarOpen: state.ui.sidebarOpen,
            darkMode: state.ui.darkMode,
          },
        }),
      }
    ),
    {
      name: 'wallet-dashboard-store',
    }
  )

