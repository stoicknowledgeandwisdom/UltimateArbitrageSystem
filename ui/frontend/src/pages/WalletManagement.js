import React, { useState, useEffect } from 'react';
import styled from 'styled-components';

// Components (would be imported from actual component files)
const Card = styled.div`
  background-color: white;
  border-radius: 8px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
  padding: 20px;
  margin-bottom: 24px;
  
  h2 {
    margin-top: 0;
    margin-bottom: 16px;
    font-size: 1.2rem;
    font-weight: 600;
    color: #1e3a8a;
  }
`;

const TabsContainer = styled.div`
  display: flex;
  margin-bottom: 20px;
  border-bottom: 1px solid #e5e7eb;
`;

const TabButton = styled.button`
  background: ${props => props.active ? '#1e40af' : 'transparent'};
  color: ${props => props.active ? 'white' : '#4b5563'};
  border: none;
  padding: 12px 24px;
  font-size: 1rem;
  font-weight: 500;
  cursor: pointer;
  transition: all 0.2s;
  border-radius: 6px 6px 0 0;
  
  &:hover {
    background: ${props => props.active ? '#1e40af' : '#f3f4f6'};
  }
`;

const ExchangeSelector = styled.div`
  margin-bottom: 20px;
  
  select {
    padding: 8px 12px;
    border-radius: 4px;
    border: 1px solid #d1d5db;
    font-size: 1rem;
    width: 200px;
  }
`;

const PageHeader = styled.div`
  margin-bottom: 24px;
  
  h1 {
    font-size: 1.8rem;
    font-weight: 700;
    color: #111827;
    margin-bottom: 8px;
  }
  
  p {
    color: #6b7280;
    margin: 0;
  }
`;

// Mock component for balance table
const BalanceTable = () => (
  <table style={{ width: '100%', borderCollapse: 'collapse' }}>
    <thead>
      <tr>
        <th style={{ textAlign: 'left', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>Asset</th>
        <th style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>Available</th>
        <th style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>In Order</th>
        <th style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>Total</th>
        <th style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>USD Value</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td style={{ padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>BTC</td>
        <td style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>0.5432</td>
        <td style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>0.1000</td>
        <td style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>0.6432</td>
        <td style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>$32,160.00</td>
      </tr>
      <tr>
        <td style={{ padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>ETH</td>
        <td style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>2.7500</td>
        <td style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>0.5000</td>
        <td style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>3.2500</td>
        <td style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>$7,150.00</td>
      </tr>
      <tr>
        <td style={{ padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>USDT</td>
        <td style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>15,432.25</td>
        <td style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>2,500.00</td>
        <td style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>17,932.25</td>
        <td style={{ textAlign: 'right', padding: '12px 8px', borderBottom: '1px solid #e5e7eb' }}>$17,932.25</td>
      </tr>
    </tbody>
  </table>
);

// Mock component for deposit/withdraw form
const DepositWithdrawForm = ({ type }) => (
  <form style={{ maxWidth: '500px' }}>
    <div style={{ marginBottom: '16px' }}>
      <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>Asset</label>
      <select style={{ width: '100%', padding: '10px', borderRadius: '4px', border: '1px solid #d1d5db' }}>
        <option value="BTC">Bitcoin (BTC)</option>
        <option value="ETH">Ethereum (ETH)</option>
        <option value="USDT">Tether (USDT)</option>
      </select>
    </div>
    
    <div style={{ marginBottom: '16px' }}>
      <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>
        {type === 'deposit' ? 'Deposit Address' : 'Withdrawal Address'}
      </label>
      <input 
        type="text" 
        style={{ width: '100%', padding: '10px', borderRadius: '4px', border: '1px solid #d1d5db' }}
        placeholder={type === 'deposit' ? 'Your deposit address will appear here' : 'Enter destination address'}
      />
    </div>
    
    {type === 'withdraw' && (
      <div style={{ marginBottom: '16px' }}>
        <label style={{ display: 'block', marginBottom: '8px', fontWeight: '500' }}>Amount</label>
        <input 
          type="

