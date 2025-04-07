import React, { useEffect } from 'react';
import { Navigate } from 'react-router-dom';
import StrategyManagement from './StrategyManagement';

/**
 * Strategies page - This component serves as a wrapper/redirect to the existing StrategyManagement page.
 * We keep this component for future expansion and to maintain consistency with the routing in App.js.
 */
const Strategies = () => {
  useEffect(() => {
    // Log for analytics or other tracking purposes
    console.log('Redirecting from Strategies to StrategyManagement page');
  }, []);

  // Option 1: Simply render the StrategyManagement component
  return <StrategyManagement />;
  
  // Option 2 (alternative): Redirect to the strategy management URL
  // return <Navigate to="/strategy-management" replace />;
};

export default Strategies;
