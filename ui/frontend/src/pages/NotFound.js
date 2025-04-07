import React from 'react';
import { 
  Box, 
  Container, 
  Typography, 
  Button, 
  Paper,
  Grid,
  Divider
} from '@mui/material';
import { 
  Home as HomeIcon,
  Dashboard as DashboardIcon,
  ShowChart as ChartIcon,
  Settings as SettingsIcon,
  Error as ErrorIcon
} from '@mui/icons-material';
import { Link as RouterLink } from 'react-router-dom';

const NotFound = () => {
  return (
    <Container maxWidth="md" sx={{ mt: 10, mb: 4 }}>
      <Paper 
        elevation={3} 
        sx={{ 
          p: 4, 
          textAlign: 'center',
          borderRadius: 2,
          backgroundColor: (theme) => 
            theme.palette.mode === 'dark' ? 'rgba(66, 66, 66, 0.7)' : 'rgba(255, 255, 255, 0.8)',
          backdropFilter: 'blur(8px)'
        }}
      >
        <Box 
          sx={{ 
            display: 'flex', 
            flexDirection: 'column', 
            alignItems: 'center',
            mb: 4
          }}
        >
          <ErrorIcon 
            color="error" 
            sx={{ 
              fontSize: 100,
              mb: 2,
              animation: 'pulse 2s infinite',
              '@keyframes pulse': {
                '0%': { opacity: 1, transform: 'scale(1)' },
                '50%': { opacity: 0.7, transform: 'scale(0.95)' },
                '100%': { opacity: 1, transform: 'scale(1)' }
              }
            }} 
          />
          <Typography variant="h2" component="h1" gutterBottom>
            404
          </Typography>
          <Typography variant="h4" gutterBottom>
            Page Not Found
          </Typography>
          <Typography variant="body1" color="textSecondary" sx={{ maxWidth: '600px', mb: 4 }}>
            The page you're looking for doesn't exist or has been moved.
            Check the URL or navigate back to a known location using the links below.
          </Typography>
          
          <Divider sx={{ width: '80%', mb: 4 }} />
          
          <Typography variant="h6" gutterBottom align="center">
            Return to:
          </Typography>
          
          <Grid container spacing={2} justifyContent="center" sx={{ maxWidth: 600, mx: 'auto' }}>
            <Grid item xs={12} sm={6} md={3}>
              <Button
                component={RouterLink}
                to="/"
                variant="contained"
                color="primary"
                startIcon={<HomeIcon />}
                fullWidth
                sx={{ py: 1.5 }}
              >
                Home
              </Button>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Button
                component={RouterLink}
                to="/dashboard"
                variant="outlined"
                startIcon={<DashboardIcon />}
                fullWidth
                sx={{ py: 1.5 }}
              >
                Dashboard
              </Button>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Button
                component={RouterLink}
                to="/trading"
                variant="outlined"
                startIcon={<ChartIcon />}
                fullWidth
                sx={{ py: 1.5 }}
              >
                Trading
              </Button>
            </Grid>
            
            <Grid item xs={12} sm={6} md={3}>
              <Button
                component={RouterLink}
                to="/settings"
                variant="outlined"
                startIcon={<SettingsIcon />}
                fullWidth
                sx={{ py: 1.5 }}
              >
                Settings
              </Button>
            </Grid>
          </Grid>
        </Box>
        
        <Box sx={{ mt: 6 }}>
          <Typography variant="body2" color="textSecondary">
            If you believe this is an error, please contact support.
          </Typography>
          <Typography variant="body2" color="textSecondary">
            Error Code: 404-PAGE-NOT-FOUND
          </Typography>
        </Box>
      </Paper>
    </Container>
  );
};

export default NotFound;

