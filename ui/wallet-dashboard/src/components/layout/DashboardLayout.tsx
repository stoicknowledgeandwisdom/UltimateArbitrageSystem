'use client'

import { ReactNode, useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import {
  Bars3Icon,
  XMarkIcon,
  WalletIcon,
  ChartBarIcon,
  ArrowRightLeftIcon,
  DocumentTextIcon,
  CogIcon,
  PlayIcon,
  ShieldCheckIcon,
  SunIcon,
  MoonIcon,
} from '@heroicons/react/24/outline'
import { useTheme } from 'next-themes'
import { UserRole } from '@/types'
import { useStore } from '@/stores/useStore'

interface DashboardLayoutProps {
  children: ReactNode
  activeView: string
  onViewChange: (view: string) => void
  user: {
    id: string
    role: UserRole
    preferences: any
  } | null
}

const navigationItems = [
  { id: 'overview', label: 'Overview', icon: ChartBarIcon, roles: ['trader', 'quant', 'compliance_officer', 'admin'] },
  { id: 'wallets', label: 'Wallets', icon: WalletIcon, roles: ['trader', 'quant', 'admin'] },
  { id: 'transfers', label: 'Transfers', icon: ArrowRightLeftIcon, roles: ['trader', 'admin'] },
  { id: 'compliance', label: 'Compliance', icon: DocumentTextIcon, roles: ['compliance_officer', 'admin'] },
  { id: 'security', label: 'Security', icon: ShieldCheckIcon, roles: ['admin'] },
  { id: 'simulator', label: 'Simulator', icon: PlayIcon, roles: ['quant', 'admin'] },
  { id: 'settings', label: 'Settings', icon: CogIcon, roles: ['trader', 'quant', 'compliance_officer', 'admin'] },
]

export function DashboardLayout({ children, activeView, onViewChange, user }: DashboardLayoutProps) {
  const [sidebarOpen, setSidebarOpen] = useState(true)
  const { theme, setTheme } = useTheme()
  const { setUIState } = useStore()

  const filteredNavigationItems = navigationItems.filter(item => 
    user?.role?.name && item.roles.includes(user.role.name)
  )

  const toggleSidebar = () => {
    setSidebarOpen(!sidebarOpen)
    setUIState({ sidebarOpen: !sidebarOpen })
  }

  const toggleTheme = () => {
    setTheme(theme === 'dark' ? 'light' : 'dark')
  }

  return (
    <div className="h-screen flex bg-gray-50 dark:bg-gray-900">
      {/* Sidebar */}
      <AnimatePresence>
        {sidebarOpen && (
          <motion.div
            initial={{ x: -280 }}
            animate={{ x: 0 }}
            exit={{ x: -280 }}
            transition={{ duration: 0.3, ease: 'easeInOut' }}
            className="w-70 bg-white dark:bg-gray-800 shadow-lg border-r border-gray-200 dark:border-gray-700 flex flex-col"
          >
            {/* Sidebar Header */}
            <div className="p-6 border-b border-gray-200 dark:border-gray-700">
              <div className="flex items-center space-x-3">
                <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-primary-700 rounded-lg flex items-center justify-center">
                  <WalletIcon className="w-6 h-6 text-white" />
                </div>
                <div>
                  <h1 className="text-xl font-bold text-gray-900 dark:text-white">
                    Wallet Dashboard
                  </h1>
                  <p className="text-sm text-gray-500 dark:text-gray-400">
                    {user?.role?.name || 'trader'}
                  </p>
                </div>
              </div>
            </div>

            {/* Navigation */}
            <nav className="flex-1 p-4 space-y-2">
              {filteredNavigationItems.map((item) => {
                const Icon = item.icon
                const isActive = activeView === item.id
                
                return (
                  <motion.button
                    key={item.id}
                    onClick={() => onViewChange(item.id)}
                    whileHover={{ scale: 1.02 }}
                    whileTap={{ scale: 0.98 }}
                    className={`w-full flex items-center space-x-3 px-4 py-3 rounded-lg text-left transition-colors duration-200 ${
                      isActive
                        ? 'bg-primary-100 dark:bg-primary-900/50 text-primary-900 dark:text-primary-100 border border-primary-200 dark:border-primary-800'
                        : 'text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-700'
                    }`}
                  >
                    <Icon className={`w-5 h-5 ${
                      isActive ? 'text-primary-600 dark:text-primary-400' : 'text-gray-500 dark:text-gray-400'
                    }`} />
                    <span className="font-medium">{item.label}</span>
                    {isActive && (
                      <motion.div
                        layoutId="activeIndicator"
                        className="ml-auto w-2 h-2 bg-primary-600 dark:bg-primary-400 rounded-full"
                      />
                    )}
                  </motion.button>
                )
              })}
            </nav>

            {/* Sidebar Footer */}
            <div className="p-4 border-t border-gray-200 dark:border-gray-700">
              <div className="flex items-center justify-between">
                <div className="flex items-center space-x-2">
                  <div className="w-8 h-8 bg-gray-300 dark:bg-gray-600 rounded-full" />
                  <div>
                    <p className="text-sm font-medium text-gray-900 dark:text-white">
                      {user?.id || 'Demo User'}
                    </p>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Online
                    </p>
                  </div>
                </div>
                <button
                  onClick={toggleTheme}
                  className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
                >
                  {theme === 'dark' ? (
                    <SunIcon className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                  ) : (
                    <MoonIcon className="w-5 h-5 text-gray-500 dark:text-gray-400" />
                  )}
                </button>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* Main Content */}
      <div className="flex-1 flex flex-col">
        {/* Top Bar */}
        <header className="bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <button
                onClick={toggleSidebar}
                className="p-2 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                {sidebarOpen ? (
                  <XMarkIcon className="w-6 h-6 text-gray-600 dark:text-gray-400" />
                ) : (
                  <Bars3Icon className="w-6 h-6 text-gray-600 dark:text-gray-400" />
                )}
              </button>
              
              <div>
                <h2 className="text-2xl font-bold text-gray-900 dark:text-white capitalize">
                  {activeView}
                </h2>
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  {new Date().toLocaleDateString('en-US', {
                    weekday: 'long',
                    year: 'numeric',
                    month: 'long',
                    day: 'numeric'
                  })}
                </p>
              </div>
            </div>

            <div className="flex items-center space-x-4">
              {/* Status indicators */}
              <div className="flex items-center space-x-2">
                <div className="flex items-center space-x-1">
                  <div className="w-2 h-2 bg-success-500 rounded-full animate-pulse" />
                  <span className="text-xs text-gray-500 dark:text-gray-400">Live</span>
                </div>
              </div>
              
              {/* Quick actions */}
              <div className="flex items-center space-x-2">
                <button className="btn-secondary text-sm py-1 px-3">
                  Export
                </button>
                <button className="btn-primary text-sm py-1 px-3">
                  New Transaction
                </button>
              </div>
            </div>
          </div>
        </header>

        {/* Page Content */}
        <main className="flex-1 overflow-hidden">
          <motion.div
            key={activeView}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -20 }}
            transition={{ duration: 0.3 }}
            className="h-full"
          >
            {children}
          </motion.div>
        </main>
      </div>
    </div>
  )
}

