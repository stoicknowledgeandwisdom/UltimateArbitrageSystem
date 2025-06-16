// Core wallet and trading types
export interface Wallet {
  id: string;
  name: string;
  type: 'exchange' | 'onchain' | 'cold_storage';
  address?: string;
  exchangeId?: string;
  balance: number;
  currency: string;
  status: 'active' | 'inactive' | 'error';
  lastUpdated: Date;
  apiKeyEncrypted?: string;
  hardwareWalletConnected?: boolean;
}

export interface ExchangeAPIKey {
  id: string;
  exchangeName: string;
  keyName: string;
  encryptedKey: string;
  encryptedSecret: string;
  permissions: string[];
  isActive: boolean;
  lastUsed?: Date;
  createdAt: Date;
}

export interface Position {
  id: string;
  symbol: string;
  side: 'long' | 'short';
  size: number;
  entryPrice: number;
  currentPrice: number;
  unrealizedPnL: number;
  realizedPnL: number;
  percentage: number;
  leverage?: number;
  exchange: string;
  walletId: string;
}

export interface Strategy {
  id: string;
  name: string;
  type: 'arbitrage' | 'market_making' | 'trend_following' | 'mean_reversion';
  status: 'active' | 'inactive' | 'paused' | 'error';
  config: StrategyConfig;
  performance: StrategyPerformance;
  riskMetrics: RiskMetrics;
  createdAt: Date;
  updatedAt: Date;
}

export interface StrategyConfig {
  maxPosition: number;
  stopLoss: number;
  takeProfit: number;
  riskPerTrade: number;
  maxDrawdown: number;
  parameters: Record<string, any>;
}

export interface StrategyPerformance {
  totalPnL: number;
  winRate: number;
  sharpeRatio: number;
  maxDrawdown: number;
  avgTrade: number;
  totalTrades: number;
  dailyReturns: number[];
}

export interface RiskMetrics {
  var95: number;
  var99: number;
  expectedShortfall: number;
  beta: number;
  volatility: number;
  correlation: Record<string, number>;
}

export interface Transfer {
  id: string;
  from: string;
  to: string;
  amount: number;
  currency: string;
  type: 'onchain' | 'exchange' | 'bridge';
  status: 'pending' | 'confirmed' | 'failed';
  txHash?: string;
  estimatedFees: TransferFees;
  actualFees?: TransferFees;
  createdAt: Date;
  confirmedAt?: Date;
}

export interface TransferFees {
  networkFee: number;
  bridgeFee?: number;
  exchangeFee?: number;
  totalFee: number;
}

export interface BridgeRoute {
  id: string;
  fromChain: string;
  toChain: string;
  protocol: string;
  estimatedTime: number;
  fees: TransferFees;
  slippage: number;
  security: 'high' | 'medium' | 'low';
}

export interface AuditLog {
  id: string;
  timestamp: Date;
  userId: string;
  action: string;
  resource: string;
  details: Record<string, any>;
  ipAddress: string;
  userAgent: string;
  risk: 'low' | 'medium' | 'high';
}

export interface ComplianceReport {
  id: string;
  type: 'pnl' | 'trades' | 'transfers' | 'full_audit';
  format: 'csv' | 'parquet' | 'xbrl';
  dateRange: {
    from: Date;
    to: Date;
  };
  status: 'pending' | 'generating' | 'ready' | 'failed';
  downloadUrl?: string;
  createdAt: Date;
}

export interface UserRole {
  id: string;
  name: 'trader' | 'quant' | 'compliance_officer' | 'admin';
  permissions: Permission[];
  restrictions: Restriction[];
}

export interface Permission {
  resource: string;
  actions: string[];
}

export interface Restriction {
  type: 'time' | 'amount' | 'frequency';
  value: any;
  description: string;
}

// WebSocket types
export interface WSMessage {
  type: string;
  data: any;
  timestamp: number;
}

export interface PriceUpdate {
  symbol: string;
  price: number;
  timestamp: number;
  exchange?: string;
}

export interface PortfolioUpdate {
  totalValue: number;
  totalPnL: number;
  positions: Position[];
  timestamp: number;
}

// Hardware wallet types
export interface HardwareWallet {
  type: 'ledger' | 'trezor';
  connected: boolean;
  address?: string;
  deviceId?: string;
}

export interface SigningRequest {
  id: string;
  type: 'transaction' | 'message';
  data: any;
  qrCode?: string;
  status: 'pending' | 'signed' | 'rejected';
}

// Simulator types
export interface SimulationConfig {
  id: string;
  name: string;
  strategy: string;
  startDate: Date;
  endDate: Date;
  initialCapital: number;
  parameters: Record<string, any>;
}

export interface SimulationResult {
  id: string;
  configId: string;
  totalReturn: number;
  sharpeRatio: number;
  maxDrawdown: number;
  winRate: number;
  trades: SimulatedTrade[];
  equity: EquityCurve[];
  metrics: Record<string, number>;
}

export interface SimulatedTrade {
  timestamp: Date;
  symbol: string;
  side: 'buy' | 'sell';
  quantity: number;
  price: number;
  pnl: number;
}

export interface EquityCurve {
  timestamp: Date;
  equity: number;
  drawdown: number;
}

// Chart and visualization types
export interface ChartData {
  labels: string[];
  datasets: ChartDataset[];
}

export interface ChartDataset {
  label: string;
  data: number[];
  borderColor?: string;
  backgroundColor?: string;
  fill?: boolean;
}

export interface HeatmapData {
  x: string;
  y: string;
  value: number;
  color?: string;
}

// API Response types
export interface APIResponse<T> {
  success: boolean;
  data?: T;
  error?: string;
  timestamp: number;
}

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  pageSize: number;
  hasMore: boolean;
}

// Form types
export interface WalletFormData {
  name: string;
  type: Wallet['type'];
  address?: string;
  exchangeId?: string;
  apiKey?: string;
  apiSecret?: string;
}

export interface TransferFormData {
  fromWallet: string;
  toWallet: string;
  amount: number;
  currency: string;
  type: Transfer['type'];
}

export interface StrategyFormData {
  name: string;
  type: Strategy['type'];
  config: StrategyConfig;
}

// State management types
export interface AppState {
  user: {
    id: string;
    role: UserRole;
    preferences: UserPreferences;
  };
  wallets: Wallet[];
  positions: Position[];
  strategies: Strategy[];
  transfers: Transfer[];
  auditLogs: AuditLog[];
  websocket: {
    connected: boolean;
    lastMessage?: WSMessage;
  };
  ui: {
    sidebarOpen: boolean;
    activeView: string;
    loading: boolean;
    error?: string;
  };
}

export interface UserPreferences {
  theme: 'light' | 'dark';
  currency: string;
  notifications: NotificationSettings;
  dashboard: DashboardLayout;
}

export interface NotificationSettings {
  email: boolean;
  push: boolean;
  trades: boolean;
  pnl: boolean;
  security: boolean;
}

export interface DashboardLayout {
  widgets: DashboardWidget[];
  layout: 'grid' | 'flex';
}

export interface DashboardWidget {
  id: string;
  type: 'pnl' | 'positions' | 'orders' | 'chart' | 'heatmap';
  position: { x: number; y: number; w: number; h: number };
  config: Record<string, any>;
}

