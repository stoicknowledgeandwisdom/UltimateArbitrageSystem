import React, { useState, useEffect } from 'react';
import {
  Container,
  Paper,
  Typography,
  Grid,
  TextField,
  Button,
  Switch,
  FormControlLabel,
  Divider,
  Box,
  Card,
  CardContent,
  CardActions,
  FormControl,
  InputLabel,
  Select,
  MenuItem,
  Tabs,
  Tab,
  IconButton,
  Alert,
  Snackbar,
  Dialog,
  DialogTitle,
  DialogContent,
  DialogContentText,
  DialogActions
} from '@mui/material';
import {
  Person as PersonIcon,
  VpnKey as ApiKeyIcon,
  Notifications as NotificationIcon,
  Palette as ThemeIcon,
  Security as SecurityIcon,
  Add as AddIcon,
  Visibility as VisibilityIcon,
  VisibilityOff as VisibilityOffIcon,
  Delete as DeleteIcon,
  Edit as EditIcon,
  Save as SaveIcon
} from '@mui/icons-material';

// TabPanel component for tab content
function TabPanel(props) {
  const { children, value, index, ...other } = props;

  return (
    <div
      role="tabpanel"
      hidden={value !== index}
      id={`settings-tabpanel-${index}`}
      aria-labelledby={`settings-tab-${index}`}
      {...other}
    >
      {value === index && (
        <Box sx={{ p: 3 }}>
          {children}
        </Box>
      )}
    </div>
  );
}

const Settings = () => {
  // Tab state
  const [currentTab, setCurrentTab] = useState(0);
  
  // User profile settings
  const [userProfile, setUserProfile] = useState({
    name: 'John Doe',
    email: 'john.doe@example.com',
    phone: '+1 (555) 123-4567',
    timezone: 'UTC-5',
    twoFactorEnabled: true
  });
  
  // API configuration settings
  const [apiConfig, setApiConfig] = useState({
    baseUrl: 'https://api.example.com',
    timeout: 10000,
    retryCount: 3,
    useMockData: true
  });
  
  // Notification settings
  const [notificationSettings, setNotificationSettings] = useState({
    emailNotifications: true,
    pushNotifications: false,
    alertThreshold: 0.5,
    dailySummary: true,
    opportunityAlerts: true,
    systemAlerts: true
  });
  
  // Theme settings
  const [themeSettings, setThemeSettings] = useState({
    darkMode: true,
    accentColor: 'blue',
    denseMode: false,
    animationsEnabled: true
  });
  
  // Exchange API keys
  const [apiKeys, setApiKeys] = useState([
    { id: 1, exchange: 'binance', label: 'Binance Main', apiKey: 'xxxxxxxxxxxxxxxxxxxxxxx', secretKey: 'xxxxxxxxxxxxxxxxxxxxxxx', isActive: true },
    { id: 2, exchange: 'coinbase', label: 'Coinbase Pro', apiKey: 'xxxxxxxxxxxxxxxxxxxxxxx', secretKey: 'xxxxxxxxxxxxxxxxxxxxxxx', isActive: true },
    { id: 3, exchange: 'kraken', label: 'Kraken', apiKey: 'xxxxxxxxxxxxxxxxxxxxxxx', secretKey: 'xxxxxxxxxxxxxxxxxxxxxxx', isActive: false }
  ]);
  
  // New API key state
  const [newApiKey, setNewApiKey] = useState({
    exchange: '',
    label: '',
    apiKey: '',
    secretKey: ''
  });
  
  // UI state
  const [showSecretKeys, setShowSecretKeys] = useState({});
  const [addingNewKey, setAddingNewKey] = useState(false);
  const [notification, setNotification] = useState({ open: false, message: '', severity: 'success' });
  const [confirmDialog, setConfirmDialog] = useState({ open: false, title: '', message: '', id: null });
  
  // Handle tab change
  const handleTabChange = (event, newValue) => {
    setCurrentTab(newValue);
  };
  
  // Handle user profile form changes
  const handleUserProfileChange = (event) => {
    const { name, value, checked } = event.target;
    setUserProfile({
      ...userProfile,
      [name]: name === 'twoFactorEnabled' ? checked : value
    });
  };
  
  // Handle API config form changes
  const handleApiConfigChange = (event) => {
    const { name, value, checked } = event.target;
    setApiConfig({
      ...apiConfig,
      [name]: name === 'useMockData' ? checked : (name === 'timeout' || name === 'retryCount' ? Number(value) : value)
    });
  };
  
  // Handle notification settings changes
  const handleNotificationSettingChange = (event) => {
    const { name, value, checked } = event.target;
    setNotificationSettings({
      ...notificationSettings,
      [name]: typeof checked !== 'undefined' ? checked : value
    });
  };
  
  // Handle theme settings changes
  const handleThemeSettingChange = (event) => {
    const { name, value, checked } = event.target;
    setThemeSettings({
      ...themeSettings,
      [name]: typeof checked !== 'undefined' ? checked : value
    });
  };
  
  // Toggle visibility of secret keys
  const toggleSecretKeyVisibility = (id) => {
    setShowSecretKeys({
      ...showSecretKeys,
      [id]: !showSecretKeys[id]
    });
  };
  
  // Handle new API key form changes
  const handleNewApiKeyChange = (event) => {
    const { name, value } = event.target;
    setNewApiKey({
      ...newApiKey,
      [name]: value
    });
  };
  
  // Add new API key
  const handleAddApiKey = () => {
    if (!newApiKey.exchange || !newApiKey.label || !newApiKey.apiKey || !newApiKey.secretKey) {
      setNotification({
        open: true, 
        message: 'Please fill in all fields for the new API key',
        severity: 'error'
      });
      return;
    }
    
    const newId = Math.max(...apiKeys.map(key => key.id), 0) + 1;
    const newKeyObj = {
      id: newId,
      exchange: newApiKey.exchange,
      label: newApiKey.label,
      apiKey: newApiKey.apiKey,
      secretKey: newApiKey.secretKey,
      isActive: true
    };
    
    setApiKeys([...apiKeys, newKeyObj]);
    setNewApiKey({ exchange: '', label: '', apiKey: '', secretKey: '' });
    setAddingNewKey(false);
    setNotification({
      open: true, 
      message: `Added new ${newApiKey.exchange} API key: ${newApiKey.label}`,
      severity: 'success'
    });
  };
  
  // Toggle API key active status
  const toggleApiKeyStatus = (id) => {
    setApiKeys(apiKeys.map(key => 
      key.id === id ? { ...key, isActive: !key.isActive } : key
    ));
    
    const keyInfo = apiKeys.find(key => key.id === id);
    setNotification({
      open: true, 
      message: `${keyInfo.label} ${keyInfo.isActive ? 'deactivated' : 'activated'}`,
      severity: 'info'
    });
  };
  
  // Delete API key with confirmation
  const confirmDeleteApiKey = (id) => {
    const keyInfo = apiKeys.find(key => key.id === id);
    setConfirmDialog({
      open: true,
      title: 'Delete API Key',
      message: `Are you sure you want to delete the API key for ${keyInfo.label} (${keyInfo.exchange})?`,
      id: id
    });
  };
  
  const handleDeleteApiKey = () => {
    const id = confirmDialog.id;
    const keyInfo = apiKeys.find(key => key.id === id);
    
    setApiKeys(apiKeys.filter(key => key.id !== id));
    setConfirmDialog({ open: false, title: '', message: '', id: null });
    setNotification({
      open: true, 
      message: `Deleted API key: ${keyInfo.label}`,
      severity: 'success'
    });
  };
  
  // Save settings
  const saveSettings = (settingType) => {
    // In a real app, this would send the settings to an API
    let message = 'Settings saved successfully';
    
    switch (settingType) {
      case 'profile':
        message = 'User profile updated';
        break;
      case 'api':
        message = 'API configuration updated';
        break;
      case 'notifications':
        message = 'Notification preferences saved';
        break;
      case 'theme':
        message = 'Theme settings updated';
        break;
      default:
        message = 'Settings saved successfully';
    }
    
    setNotification({
      open: true, 
      message,
      severity: 'success'
    });
  };
  
  // Close notification
  const handleCloseNotification = () => {
    setNotification({ ...notification, open: false });
  };

  return (
    <Container maxWidth="lg" sx={{ mt: 4, mb: 4 }}>
      <Typography variant="h4" gutterBottom>
        Settings
      </Typography>
      
      <Paper sx={{ width: '100%' }}>
        <Box sx={{ borderBottom: 1, borderColor: 'divider' }}>
          <Tabs 
            value={currentTab} 
            onChange={handleTabChange} 
            aria-label="settings tabs"
            variant="scrollable"
            scrollButtons="auto"
          >
            <Tab icon={<PersonIcon />} label="Profile" />
            <Tab icon={<ApiKeyIcon />} label="API Configuration" />
            <Tab icon={<NotificationIcon />} label="Notifications" />
            <Tab icon={<ThemeIcon />} label="Appearance" />
            <Tab icon={<SecurityIcon />} label="Exchange API Keys" />
          </Tabs>
        </Box>
        
        {/* User Profile Settings */}
        <TabPanel value={currentTab} index={0}>
          <Typography variant="h6" gutterBottom>
            User Profile
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Name"
                name="name"
                value={userProfile.name}
                onChange={handleUserProfileChange}
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Email"
                name="email"
                type="email"
                value={userProfile.email}
                onChange={handleUserProfileChange}
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Phone"
                name="phone"
                value={userProfile.phone}
                onChange={handleUserProfileChange}
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth margin="normal">
                <InputLabel>Timezone</InputLabel>
                <Select
                  name="timezone"
                  value={userProfile.timezone}
                  onChange={handleUserProfileChange}
                >
                  <MenuItem value="UTC-12">UTC-12</MenuItem>
                  <MenuItem value="UTC-11">UTC-11</MenuItem>
                  <MenuItem value="UTC-10">UTC-10</MenuItem>
                  <MenuItem value="UTC-9">UTC-9</MenuItem>
                  <MenuItem value="UTC-8">UTC-8 (PST)</MenuItem>
                  <MenuItem value="UTC-7">UTC-7 (MST)</MenuItem>
                  <MenuItem value="UTC-6">UTC-6 (CST)</MenuItem>
                  <MenuItem value="UTC-5">UTC-5 (EST)</MenuItem>
                  <MenuItem value="UTC-4">UTC-4</MenuItem>
                  <MenuItem value="UTC-3">UTC-3</MenuItem>
                  <MenuItem value="UTC-2">UTC-2</MenuItem>
                  <MenuItem value="UTC-1">UTC-1</MenuItem>
                  <MenuItem value="UTC+0">UTC+0</MenuItem>
                  <MenuItem value="UTC+1">UTC+1 (CET)</MenuItem>
                  <MenuItem value="UTC+2">UTC+2</MenuItem>
                  <MenuItem value="UTC+3">UTC+3</MenuItem>
                  <MenuItem value="UTC+4">UTC+4</MenuItem>
                  <MenuItem value="UTC+5">UTC+5</MenuItem>
                  <MenuItem value="UTC+6">UTC+6</MenuItem>
                  <MenuItem value="UTC+7">UTC+7</MenuItem>
                  <MenuItem value="UTC+8">UTC+8</MenuItem>
                  <MenuItem value="UTC+9">UTC+9 (JST)</MenuItem>
                  <MenuItem value="UTC+10">UTC+10</MenuItem>
                  <MenuItem value="UTC+11">UTC+11</MenuItem>
                  <MenuItem value="UTC+12">UTC+12</MenuItem>
                </Select>
              </FormControl>
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={userProfile.twoFactorEnabled}
                    onChange={handleUserProfileChange}
                    name="twoFactorEnabled"
                  />
                }
                label="Enable Two-Factor Authentication"
                label="Enable Two-Factor Authentication"
              />
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
            <Button 
              variant="contained" 
              startIcon={<SaveIcon />}
              onClick={() => saveSettings('profile')}
            >
              Save Profile
            </Button>
          </Box>
        </TabPanel>
        
        {/* API Configuration Settings */}
        <TabPanel value={currentTab} index={1}>
          <Typography variant="h6" gutterBottom>
            API Configuration
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="API Base URL"
                name="baseUrl"
                value={apiConfig.baseUrl}
                onChange={handleApiConfigChange}
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Request Timeout (ms)"
                name="timeout"
                type="number"
                value={apiConfig.timeout}
                onChange={handleApiConfigChange}
                margin="normal"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Retry Count"
                name="retryCount"
                type="number"
                value={apiConfig.retryCount}
                onChange={handleApiConfigChange}
                margin="normal"
              />
            </Grid>
            <Grid item xs={12}>
              <FormControlLabel
                control={
                  <Switch
                    checked={apiConfig.useMockData}
                    onChange={handleApiConfigChange}
                    name="useMockData"
                  />
                }
                label="Use Mock Data (for testing/development)"
              />
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
            <Button 
              variant="contained" 
              startIcon={<SaveIcon />}
              onClick={() => saveSettings('api')}
            >
              Save API Configuration
            </Button>
          </Box>
        </TabPanel>
        
        {/* Notification Settings */}
        <TabPanel value={currentTab} index={2}>
          <Typography variant="h6" gutterBottom>
            Notifications
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.emailNotifications}
                    onChange={handleNotificationSettingChange}
                    name="emailNotifications"
                  />
                }
                label="Email Notifications"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.pushNotifications}
                    onChange={handleNotificationSettingChange}
                    name="pushNotifications"
                  />
                }
                label="Push Notifications"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.dailySummary}
                    onChange={handleNotificationSettingChange}
                    name="dailySummary"
                  />
                }
                label="Daily Summary Reports"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.opportunityAlerts}
                    onChange={handleNotificationSettingChange}
                    name="opportunityAlerts"
                  />
                }
                label="Arbitrage Opportunity Alerts"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={notificationSettings.systemAlerts}
                    onChange={handleNotificationSettingChange}
                    name="systemAlerts"
                  />
                }
                label="System Alerts"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <TextField
                fullWidth
                label="Profit Alert Threshold (%)"
                name="alertThreshold"
                type="number"
                value={notificationSettings.alertThreshold * 100}
                onChange={(e) => handleNotificationSettingChange({
                  target: {
                    name: 'alertThreshold',
                    value: parseFloat(e.target.value) / 100
                  }
                })}
                margin="normal"
                InputProps={{
                  endAdornment: '%'
                }}
              />
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
            <Button 
              variant="contained" 
              startIcon={<SaveIcon />}
              onClick={() => saveSettings('notifications')}
            >
              Save Notification Settings
            </Button>
          </Box>
        </TabPanel>
        
        {/* Theme Settings */}
        <TabPanel value={currentTab} index={3}>
          <Typography variant="h6" gutterBottom>
            Appearance
          </Typography>
          
          <Grid container spacing={3}>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={themeSettings.darkMode}
                    onChange={handleThemeSettingChange}
                    name="darkMode"
                  />
                }
                label="Dark Mode"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={themeSettings.denseMode}
                    onChange={handleThemeSettingChange}
                    name="denseMode"
                  />
                }
                label="Dense Mode (compact UI)"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControlLabel
                control={
                  <Switch
                    checked={themeSettings.animationsEnabled}
                    onChange={handleThemeSettingChange}
                    name="animationsEnabled"
                  />
                }
                label="Enable Animations"
              />
            </Grid>
            <Grid item xs={12} md={6}>
              <FormControl fullWidth margin="normal">
                <InputLabel>Accent Color</InputLabel>
                <Select
                  name="accentColor"
                  value={themeSettings.accentColor}
                  onChange={handleThemeSettingChange}
                >
                  <MenuItem value="blue">Blue</MenuItem>
                  <MenuItem value="purple">Purple</MenuItem>
                  <MenuItem value="green">Green</MenuItem>
                  <MenuItem value="orange">Orange</MenuItem>
                  <MenuItem value="red">Red</MenuItem>
                </Select>
              </FormControl>
            </Grid>
          </Grid>
          
          <Box sx={{ mt: 3, display: 'flex', justifyContent: 'flex-end' }}>
            <Button 
              variant="contained" 
              startIcon={<SaveIcon />}
              onClick={() => saveSettings('theme')}
            >
              Save Appearance Settings
            </Button>
          </Box>
        </TabPanel>
        
        {/* Exchange API Keys */}
        <TabPanel value={currentTab} index={4}>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
            <Typography variant="h6">
              Exchange API Keys
            </Typography>
            <Button 
              variant="contained" 
              startIcon={<AddIcon />}
              onClick={() => setAddingNewKey(true)}
              disabled={addingNewKey}
            >
              Add New API Key
            </Button>
          </Box>
          
          {addingNewKey && (
            <Paper sx={{ p: 2, mb: 3 }}>
              <Typography variant="subtitle1" gutterBottom>
                Add New API Key
              </Typography>
              <Grid container spacing={2}>
                <Grid item xs={12} md={6}>
                  <FormControl fullWidth margin="normal">
                    <InputLabel>Exchange</InputLabel>
                    <Select
                      name="exchange"
                      value={newApiKey.exchange}
                      onChange={handleNewApiKeyChange}
                    >
                      <MenuItem value="binance">Binance</MenuItem>
                      <MenuItem value="coinbase">Coinbase</MenuItem>
                      <MenuItem value="kraken">Kraken</MenuItem>
                      <MenuItem value="kucoin">KuCoin</MenuItem>
                      <MenuItem value="huobi">Huobi</MenuItem>
                      <MenuItem value="bitfinex">Bitfinex</MenuItem>
                    </Select>
                  </FormControl>
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Label"
                    name="label"
                    value={newApiKey.label}
                    onChange={handleNewApiKeyChange}
                    margin="normal"
                    placeholder="e.g., Binance Main Account"
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="API Key"
                    name="apiKey"
                    value={newApiKey.apiKey}
                    onChange={handleNewApiKeyChange}
                    margin="normal"
                  />
                </Grid>
                <Grid item xs={12} md={6}>
                  <TextField
                    fullWidth
                    label="Secret Key"
                    name="secretKey"
                    type="password"
                    value={newApiKey.secretKey}
                    onChange={handleNewApiKeyChange}
                    margin="normal"
                  />
                </Grid>
              </Grid>
              <Box sx={{ mt: 2, display: 'flex', justifyContent: 'flex-end', gap: 1 }}>
                <Button 
                  variant="outlined" 
                  onClick={() => {
                    setAddingNewKey(false);
                    setNewApiKey({ exchange: '', label: '', apiKey: '', secretKey: '' });
                  }}
                >
                  Cancel
                </Button>
                <Button 
                  variant="contained" 
                  onClick={handleAddApiKey}
                >
                  Add API Key
                </Button>
              </Box>
            </Paper>
          )}
          
          {apiKeys.length === 0 ? (
            <Typography variant="body1" color="textSecondary" sx={{ mt: 2 }}>
              No API keys have been added yet. Add your first exchange API key to start trading.
            </Typography>
          ) : (
            <Grid container spacing={3}>
              {apiKeys.map((key) => (
                <Grid item xs={12} md={6} key={key.id}>
                  <Card>
                    <CardContent>
                      <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 2 }}>
                        <Typography variant="h6">
                          {key.label}
                        </Typography>
                        <Chip 
                          label={key.isActive ? 'Active' : 'Inactive'} 
                          color={key.isActive ? 'success' : 'default'}
                          size="small"
                        />
                      </Box>
                      <Typography variant="body2" color="textSecondary" gutterBottom>
                        Exchange: {key.exchange.charAt(0).toUpperCase() + key.exchange.slice(1)}
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        API Key:
                      </Typography>
                      <Typography variant="body2" sx={{ fontFamily: 'monospace', wordBreak: 'break-all' }}>
                        {key.apiKey.substring(0, 4)}...{key.apiKey.substring(key.apiKey.length - 4)}
                      </Typography>
                      <Typography variant="body2" sx={{ mt: 1 }}>
                        Secret Key:
                      </Typography>
                      <Box sx={{ display: 'flex', alignItems: 'center' }}>
                        <Typography 
                          variant="body2" 
                          sx={{ 
                            fontFamily: 'monospace', 
                            wordBreak: 'break-all',
                            flexGrow: 1
                          }}
                        >
                          {showSecretKeys[key.id] 
                            ? key.secretKey 
                            : '••••••••••••••••••••••••••'}
                        </Typography>
                        <IconButton 
                          size="small" 
                          onClick={() => toggleSecretKeyVisibility(key.id)}
                          aria-label="toggle secret key visibility"
                        >
                          {showSecretKeys[key.id] ? <VisibilityOffIcon /> : <VisibilityIcon />}
                        </IconButton>
                      </Box>
                    </CardContent>
                    <CardActions>
                      <Button
                        size="small"
                        startIcon={key.isActive ? null : <AddIcon />}
                        color={key.isActive ? "error" : "success"}
                        onClick={() => toggleApiKeyStatus(key.id)}
                      >
                        {key.isActive ? 'Deactivate' : 'Activate'}
                      </Button>
                      <Button
                        size="small"
                        startIcon={<DeleteIcon />}
                        onClick={() => confirmDeleteApiKey(key.id)}
                      >
                        Delete
                      </Button>
                    </CardActions>
                  </Card>
                </Grid>
              ))}
            </Grid>
          )}
        </TabPanel>
      </Paper>
      
      {/* Confirmation Dialog */}
      <Dialog
        open={confirmDialog.open}
        onClose={() => setConfirmDialog({ ...confirmDialog, open: false })}
        aria-labelledby="alert-dialog-title"
        aria-describedby="alert-dialog-description"
      >
        <DialogTitle id="alert-dialog-title">
          {confirmDialog.title}
        </DialogTitle>
        <DialogContent>
          <DialogContentText id="alert-dialog-description">
            {confirmDialog.message}
          </DialogContentText>
        </DialogContent>
        <DialogActions>
          <Button onClick={() => setConfirmDialog({ ...confirmDialog, open: false })}>
            Cancel
          </Button>
          <Button onClick={handleDeleteApiKey} autoFocus color="error">
            Delete
          </Button>
        </DialogActions>
      </Dialog>
      
      {/* Notifications */}
      <Snackbar
        open={notification.open}
        autoHideDuration={6000}
        onClose={handleCloseNotification}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'right' }}
      >
        <Alert 
          onClose={handleCloseNotification} 
          severity={notification.severity} 
          sx={{ width: '100%' }}
        >
          {notification.message}
        </Alert>
      </Snackbar>
    </Container>
  );
};

export default Settings;
