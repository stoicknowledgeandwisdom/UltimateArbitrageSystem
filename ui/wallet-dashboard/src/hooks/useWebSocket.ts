import { useEffect, useRef, useState, useCallback } from 'react'
import { useStore } from '@/stores/useStore'
import { WSMessage, PriceUpdate, PortfolioUpdate } from '@/types'
import toast from 'react-hot-toast'

interface UseWebSocketReturn {
  connected: boolean
  send: (message: any) => void
  disconnect: () => void
  reconnect: () => void
}

export function useWebSocket(url: string): UseWebSocketReturn {
  const wsRef = useRef<WebSocket | null>(null)
  const [connected, setConnected] = useState(false)
  const reconnectTimeoutRef = useRef<NodeJS.Timeout>()
  const reconnectAttemptsRef = useRef(0)
  const maxReconnectAttempts = 5
  const reconnectInterval = 5000
  
  const { setWebSocketState, updatePosition, updateWallet } = useStore()
  
  const handleMessage = useCallback((event: MessageEvent) => {
    try {
      const message: WSMessage = JSON.parse(event.data)
      
      // Update store with last message
      setWebSocketState({ lastMessage: message })
      
      // Handle different message types
      switch (message.type) {
        case 'price_update':
          const priceUpdate = message.data as PriceUpdate
          // Update positions with new prices
          useStore.getState().positions.forEach(position => {
            if (position.symbol === priceUpdate.symbol) {
              const newUnrealizedPnL = 
                (priceUpdate.price - position.entryPrice) * position.size *
                (position.side === 'long' ? 1 : -1)
              
              updatePosition(position.id, {
                currentPrice: priceUpdate.price,
                unrealizedPnL: newUnrealizedPnL,
                percentage: (newUnrealizedPnL / (position.entryPrice * position.size)) * 100
              })
            }
          })
          break
          
        case 'portfolio_update':
          const portfolioUpdate = message.data as PortfolioUpdate
          portfolioUpdate.positions.forEach(position => {
            updatePosition(position.id, position)
          })
          break
          
        case 'wallet_balance_update':
          const { walletId, balance } = message.data
          updateWallet(walletId, { balance, lastUpdated: new Date() })
          break
          
        case 'strategy_alert':
          const { strategy, alert } = message.data
          toast.error(`Strategy Alert: ${strategy} - ${alert}`)
          break
          
        case 'security_alert':
          const { type, details } = message.data
          toast.error(`Security Alert: ${type} - ${details}`)
          break
          
        case 'system_notification':
          const { level, message: notificationMessage } = message.data
          if (level === 'error') {
            toast.error(notificationMessage)
          } else if (level === 'warning') {
            toast(notificationMessage, { icon: '⚠️' })
          } else {
            toast.success(notificationMessage)
          }
          break
          
        default:
          console.log('Unknown message type:', message.type)
      }
    } catch (error) {
      console.error('Error parsing WebSocket message:', error)
    }
  }, [setWebSocketState, updatePosition, updateWallet])
  
  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      return
    }
    
    try {
      wsRef.current = new WebSocket(url)
      
      wsRef.current.onopen = () => {
        console.log('WebSocket connected')
        setConnected(true)
        setWebSocketState({ connected: true, reconnectAttempts: 0 })
        reconnectAttemptsRef.current = 0
        
        // Send authentication/subscription message
        const authMessage = {
          type: 'auth',
          data: {
            userId: useStore.getState().user?.id,
            token: localStorage.getItem('auth_token')
          }
        }
        wsRef.current?.send(JSON.stringify(authMessage))
        
        toast.success('Connected to live data feed')
      }
      
      wsRef.current.onmessage = handleMessage
      
      wsRef.current.onclose = (event) => {
        console.log('WebSocket disconnected:', event.code, event.reason)
        setConnected(false)
        setWebSocketState({ connected: false })
        
        // Only show disconnect toast if it wasn't a manual disconnect
        if (event.code !== 1000) {
          toast.error('Disconnected from live data feed')
        }
        
        // Attempt to reconnect if not a manual close
        if (event.code !== 1000 && reconnectAttemptsRef.current < maxReconnectAttempts) {
          reconnectAttemptsRef.current++
          setWebSocketState({ reconnectAttempts: reconnectAttemptsRef.current })
          
          reconnectTimeoutRef.current = setTimeout(() => {
            console.log(`Reconnection attempt ${reconnectAttemptsRef.current}/${maxReconnectAttempts}`)
            connect()
          }, reconnectInterval)
        } else if (reconnectAttemptsRef.current >= maxReconnectAttempts) {
          toast.error('Failed to reconnect after multiple attempts')
        }
      }
      
      wsRef.current.onerror = (error) => {
        console.error('WebSocket error:', error)
        toast.error('WebSocket connection error')
      }
    } catch (error) {
      console.error('Failed to create WebSocket connection:', error)
      toast.error('Failed to connect to live data feed')
    }
  }, [url, handleMessage, setWebSocketState])
  
  const disconnect = useCallback(() => {
    if (reconnectTimeoutRef.current) {
      clearTimeout(reconnectTimeoutRef.current)
    }
    
    if (wsRef.current) {
      wsRef.current.close(1000, 'Manual disconnect')
      wsRef.current = null
    }
    
    setConnected(false)
    setWebSocketState({ connected: false })
  }, [setWebSocketState])
  
  const send = useCallback((message: any) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(message))
    } else {
      console.warn('WebSocket is not connected')
      toast.error('Cannot send message: WebSocket not connected')
    }
  }, [])
  
  const reconnect = useCallback(() => {
    disconnect()
    reconnectAttemptsRef.current = 0
    setTimeout(connect, 1000)
  }, [connect, disconnect])
  
  useEffect(() => {
    connect()
    
    return () => {
      disconnect()
    }
  }, [connect, disconnect])
  
  return {
    connected,
    send,
    disconnect,
    reconnect
  }
}

