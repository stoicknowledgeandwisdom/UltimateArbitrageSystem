#!/usr/bin/env python3

import time
import sqlite3
import os
from datetime import datetime

def check_system_status():
    print('🚀 ULTIMATE ARBITRAGE SYSTEM - REAL-TIME STATUS')
    print('=' * 60)
    
    # Check if database exists
    db_file = 'arbitrage_opportunities.db'
    if os.path.exists(db_file):
        try:
            conn = sqlite3.connect(db_file)
            cursor = conn.cursor()
            
            # Check if table exists
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='arbitrage_opportunities'")
            table_exists = cursor.fetchone() is not None
            
            if table_exists:
                cursor.execute('SELECT COUNT(*) FROM arbitrage_opportunities')
                total_opportunities = cursor.fetchone()[0]
                print(f'💰 Total Arbitrage Opportunities Found: {total_opportunities}')
                
                if total_opportunities > 0:
                    cursor.execute('SELECT MAX(profit_usd) FROM arbitrage_opportunities')
                    max_profit = cursor.fetchone()[0]
                    print(f'🎯 Maximum Single Profit Found: ${max_profit:.2f}')
                    
                    cursor.execute('SELECT SUM(profit_usd) FROM arbitrage_opportunities')
                    total_profit = cursor.fetchone()[0]
                    print(f'💎 Total Potential Profit: ${total_profit:.2f}')
                    
                    # Get recent opportunities
                    cursor.execute('''
                        SELECT asset, buy_exchange, sell_exchange, profit_usd, profit_percentage
                        FROM arbitrage_opportunities 
                        ORDER BY timestamp DESC 
                        LIMIT 5
                    ''')
                    recent = cursor.fetchall()
                    print('\n📈 Most Recent Opportunities:')
                    for opp in recent:
                        print(f'   {opp[0]}: Buy {opp[1]} -> Sell {opp[2]} = ${opp[3]:.2f} ({opp[4]:.3%})')
                else:
                    print('⏳ No arbitrage opportunities detected yet')
            else:
                print('📊 Database table not yet created - system still in data collection phase')
                
            conn.close()
            
        except Exception as e:
            print(f'📊 Database error: {str(e)}')
    else:
        print('📊 Database not yet created - system initializing')
    
    # Check log file for recent activity
    log_files = [f for f in os.listdir('.') if f.startswith('maximum_income_test_') and f.endswith('.log')]
    if log_files:
        latest_log = max(log_files, key=lambda x: os.path.getmtime(x))
        mod_time = datetime.fromtimestamp(os.path.getmtime(latest_log))
        time_diff = datetime.now() - mod_time
        
        if time_diff.total_seconds() < 60:
            print(f'🟢 System Active: Last activity {int(time_diff.total_seconds())} seconds ago')
        else:
            print(f'🟡 System Status: Last activity {int(time_diff.total_seconds()/60)} minutes ago')
    
    print(f'⏰ Current Time: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}')
    print('🔥 System Status: HUNTING FOR MAXIMUM PROFITS WITH ZERO INVESTMENT MINDSET')
    print('=' * 60)

if __name__ == '__main__':
    check_system_status()

