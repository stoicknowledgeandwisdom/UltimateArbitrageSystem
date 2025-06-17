#!/usr/bin/env python3

import sqlite3
import os
from datetime import datetime
import json

def analyze_maximum_income_results():
    print('🔍 ANALYZING MAXIMUM INCOME SYSTEM RESULTS')
    print('=' * 60)
    
    # Find the results database
    db_files = [f for f in os.listdir('.') if f.startswith('maximum_income_results_') and f.endswith('.db')]
    
    if not db_files:
        print('No results database found')
        return
    
    # Use the most recent database
    latest_db = max(db_files, key=lambda x: os.path.getmtime(x))
    print(f'Analyzing database: {latest_db}')
    print(f'Database size: {os.path.getsize(latest_db) / 1024 / 1024:.2f} MB')
    
    try:
        conn = sqlite3.connect(latest_db)
        cursor = conn.cursor()
        
        # Get all tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        print(f'\nTables found: {len(tables)}')
        for table in tables:
            table_name = table[0]
            print(f'\n📊 Table: {table_name}')
            
            # Get table info
            cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
            count = cursor.fetchone()[0]
            print(f'   Records: {count:,}')
            
            if count > 0:
                # Get column info
                cursor.execute(f'PRAGMA table_info({table_name})')
                columns = cursor.fetchall()
                print(f'   Columns: {", ".join([col[1] for col in columns])}')
                
                # Show sample data
                cursor.execute(f'SELECT * FROM {table_name} LIMIT 3')
                sample_rows = cursor.fetchall()
                print('   Sample data:')
                for i, row in enumerate(sample_rows):
                    print(f'     Row {i+1}: {row}')
                
                # If this looks like price data, analyze it
                column_names = [col[1] for col in columns]
                if 'price' in column_names or 'bid' in column_names or 'ask' in column_names:
                    print(f'   💹 Price data analysis:')
                    if 'price' in column_names:
                        cursor.execute(f'SELECT MIN(price), MAX(price), AVG(price) FROM {table_name}')
                        min_price, max_price, avg_price = cursor.fetchone()
                        print(f'      Price range: ${min_price:.6f} - ${max_price:.6f} (avg: ${avg_price:.6f})')
                    
                    if 'exchange' in column_names:
                        cursor.execute(f'SELECT exchange, COUNT(*) FROM {table_name} GROUP BY exchange')
                        exchanges = cursor.fetchall()
                        print(f'      Exchanges: {", ".join([f"{ex[0]}({ex[1]})" for ex in exchanges])}')
                    
                    if 'symbol' in column_names:
                        cursor.execute(f'SELECT symbol, COUNT(*) FROM {table_name} GROUP BY symbol')
                        symbols = cursor.fetchall()
                        print(f'      Symbols: {", ".join([f"{sym[0]}({sym[1]})" for sym in symbols[:10]])}')
                        if len(symbols) > 10:
                            print(f'      ... and {len(symbols) - 10} more symbols')
        
        conn.close()
        
    except Exception as e:
        print(f'Error analyzing database: {str(e)}')
    
    print('\n' + '=' * 60)

def analyze_arbitrage_opportunities():
    print('\n🔍 ANALYZING ARBITRAGE OPPORTUNITIES')
    print('=' * 60)
    
    db_file = 'arbitrage_opportunities.db'
    if not os.path.exists(db_file) or os.path.getsize(db_file) == 0:
        print('No arbitrage opportunities database found or empty')
        return
    
    try:
        conn = sqlite3.connect(db_file)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
        tables = cursor.fetchall()
        
        if not tables:
            print('No tables found in arbitrage database')
            return
        
        table_name = tables[0][0]
        cursor.execute(f'SELECT COUNT(*) FROM {table_name}')
        count = cursor.fetchone()[0]
        
        if count == 0:
            print('No arbitrage opportunities recorded yet')
        else:
            print(f'Total arbitrage opportunities: {count}')
            
            # Get best opportunities
            cursor.execute(f'SELECT * FROM {table_name} ORDER BY profit_usd DESC LIMIT 10')
            opportunities = cursor.fetchall()
            
            print('\n🎆 Top 10 Arbitrage Opportunities:')
            for i, opp in enumerate(opportunities, 1):
                print(f'{i:2d}. {opp[1]} | {opp[2]} -> {opp[3]} | ${opp[4]:.2f} ({opp[5]:.3%})')
        
        conn.close()
        
    except Exception as e:
        print(f'Error analyzing arbitrage opportunities: {str(e)}')

if __name__ == '__main__':
    analyze_maximum_income_results()
    analyze_arbitrage_opportunities()

