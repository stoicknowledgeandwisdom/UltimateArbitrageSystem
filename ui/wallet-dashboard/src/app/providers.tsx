'use client'

import { ThemeProvider } from 'next-themes'
import { Toaster } from 'react-hot-toast'
import { ErrorBoundary } from 'react-error-boundary'
import { ReactNode } from 'react'

function ErrorFallback({ error, resetErrorBoundary }: { error: Error; resetErrorBoundary: () => void }) {
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
      <div className="text-center">
        <h2 className="text-2xl font-bold text-gray-900 dark:text-white mb-4">
          Something went wrong
        </h2>
        <p className="text-gray-600 dark:text-gray-400 mb-6">
          {error.message}
        </p>
        <button
          onClick={resetErrorBoundary}
          className="btn-primary"
        >
          Try again
        </button>
      </div>
    </div>
  )
}

export function Providers({ children }: { children: ReactNode }) {
  return (
    <ErrorBoundary FallbackComponent={ErrorFallback}>
      <ThemeProvider
        attribute="class"
        defaultTheme="dark"
        enableSystem
        disableTransitionOnChange
      >
        {children}
        <Toaster
          position="top-right"
          toastOptions={{
            duration: 4000,
            style: {
              background: 'var(--background)',
              color: 'var(--foreground)',
              border: '1px solid var(--border)',
            },
            success: {
              iconTheme: {
                primary: 'rgb(34 197 94)',
                secondary: 'white',
              },
            },
            error: {
              iconTheme: {
                primary: 'rgb(239 68 68)',
                secondary: 'white',
              },
            },
          }}
        />
      </ThemeProvider>
    </ErrorBoundary>
  )
}

