#!/usr/bin/env python3
"""
Data Viewer for QuantumSentiment Trading Bot

View downloaded data from the SQLite database:
- Market data (OHLCV)
- Sentiment data
- Features
- Performance metrics

Usage:
    python scripts/view_data.py --table market_data --symbol AAPL
    python scripts/view_data.py --table sentiment_data --limit 10
    python scripts/view_data.py --tables  # List all tables
"""

import argparse
import sqlite3
import pandas as pd
import sys
from pathlib import Path
from datetime import datetime

def get_database_path():
    """Get path to the SQLite database"""
    db_path = Path(__file__).parent.parent / 'quantum_sentiment.db'
    if not db_path.exists():
        print(f"❌ Database not found at: {db_path}")
        print("Run the download script first: python scripts/download_historical_data.py")
        return None
    return str(db_path)

def list_tables(db_path):
    """List all tables in the database"""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = cursor.fetchall()
        
        print("📊 Available Tables:")
        print("=" * 50)
        
        for table in tables:
            table_name = table[0]
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            count = cursor.fetchone()[0]
            print(f"  📋 {table_name}: {count:,} records")
        
        conn.close()
        return [table[0] for table in tables]
        
    except Exception as e:
        print(f"❌ Error listing tables: {e}")
        return []

def view_table(db_path, table_name, symbol=None, limit=10):
    """View data from a specific table"""
    try:
        conn = sqlite3.connect(db_path)
        
        # Build query
        query = f"SELECT * FROM {table_name}"
        params = []
        
        if symbol:
            query += " WHERE symbol = ?"
            params.append(symbol.upper())
        
        query += f" ORDER BY timestamp DESC LIMIT {limit}"
        
        # Execute query
        df = pd.read_sql_query(query, conn, params=params)
        
        if df.empty:
            print(f"📭 No data found in table '{table_name}'")
            if symbol:
                print(f"   for symbol '{symbol}'")
            return
        
        print(f"📊 Data from '{table_name}' table")
        if symbol:
            print(f"🎯 Symbol: {symbol}")
        print(f"📈 Showing {len(df)} records (latest first)")
        print("=" * 80)
        
        # Format display based on table type
        if table_name == 'market_data':
            display_market_data(df)
        elif table_name == 'sentiment_data':
            display_sentiment_data(df)
        elif table_name == 'feature_data':
            display_feature_data(df)
        else:
            print(df.to_string(index=False))
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error viewing table: {e}")

def display_market_data(df):
    """Display market data in a formatted way"""
    print(f"{'Date':<12} {'Symbol':<8} {'Open':<8} {'High':<8} {'Low':<8} {'Close':<8} {'Volume':<12} {'Timeframe':<10}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        timestamp = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d')
        print(f"{timestamp:<12} {row['symbol']:<8} {row['open']:<8.2f} {row['high']:<8.2f} "
              f"{row['low']:<8.2f} {row['close']:<8.2f} {row['volume']:<12.0f} {row['timeframe']:<10}")

def display_sentiment_data(df):
    """Display sentiment data in a formatted way"""
    print(f"{'Date':<12} {'Symbol':<8} {'Source':<12} {'Sentiment':<10} {'Confidence':<10} {'Mentions':<8}")
    print("-" * 70)
    
    for _, row in df.iterrows():
        timestamp = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d')
        sentiment = f"{row['sentiment_score']:.3f}"
        confidence = f"{row['confidence']:.3f}"
        mentions = int(row['mention_count']) if pd.notna(row['mention_count']) else 0
        
        print(f"{timestamp:<12} {row['symbol']:<8} {row['source']:<12} {sentiment:<10} {confidence:<10} {mentions:<8}")

def display_feature_data(df):
    """Display feature data in a formatted way"""
    print(f"{'Date':<12} {'Symbol':<8} {'Features':<60}")
    print("-" * 80)
    
    for _, row in df.iterrows():
        timestamp = pd.to_datetime(row['timestamp']).strftime('%Y-%m-%d')
        
        # Parse features (assuming JSON format)
        try:
            import json
            features = json.loads(row['features']) if isinstance(row['features'], str) else row['features']
            feature_summary = f"{len(features)} features: {list(features.keys())[:3]}..."
        except:
            feature_summary = str(row['features'])[:60]
        
        print(f"{timestamp:<12} {row['symbol']:<8} {feature_summary:<60}")

def get_data_summary(db_path):
    """Get summary statistics of all data"""
    try:
        conn = sqlite3.connect(db_path)
        
        print("📊 DATA SUMMARY")
        print("=" * 50)
        
        # Market data summary
        query = """
        SELECT 
            symbol,
            COUNT(*) as records,
            MIN(timestamp) as first_date,
            MAX(timestamp) as last_date,
            AVG(close) as avg_price
        FROM market_data 
        GROUP BY symbol
        ORDER BY records DESC
        """
        
        df = pd.read_sql_query(query, conn)
        if not df.empty:
            print("\n📈 Market Data:")
            for _, row in df.iterrows():
                first_date = pd.to_datetime(row['first_date']).strftime('%Y-%m-%d')
                last_date = pd.to_datetime(row['last_date']).strftime('%Y-%m-%d')
                print(f"  {row['symbol']}: {row['records']} records ({first_date} to {last_date}) - Avg: ${row['avg_price']:.2f}")
        
        # Sentiment data summary
        query = """
        SELECT 
            symbol,
            source,
            COUNT(*) as records,
            AVG(sentiment_score) as avg_sentiment
        FROM sentiment_data 
        GROUP BY symbol, source
        ORDER BY symbol, records DESC
        """
        
        df = pd.read_sql_query(query, conn)
        if not df.empty:
            print("\n💭 Sentiment Data:")
            for _, row in df.iterrows():
                print(f"  {row['symbol']} ({row['source']}): {row['records']} records - Avg sentiment: {row['avg_sentiment']:.3f}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ Error getting summary: {e}")

def main():
    parser = argparse.ArgumentParser(description="View downloaded trading data")
    
    parser.add_argument('--tables', action='store_true', help='List all available tables')
    parser.add_argument('--table', type=str, help='Table name to view')
    parser.add_argument('--symbol', type=str, help='Filter by symbol (e.g., AAPL)')
    parser.add_argument('--limit', type=int, default=10, help='Limit number of records (default: 10)')
    parser.add_argument('--summary', action='store_true', help='Show data summary')
    
    args = parser.parse_args()
    
    # Get database path
    db_path = get_database_path()
    if not db_path:
        return
    
    print(f"📁 Database: {db_path}")
    print(f"📊 Size: {Path(db_path).stat().st_size / 1024:.1f} KB")
    print()
    
    # List tables
    if args.tables:
        list_tables(db_path)
        return
    
    # Show summary
    if args.summary:
        get_data_summary(db_path)
        return
    
    # View specific table
    if args.table:
        view_table(db_path, args.table, args.symbol, args.limit)
        return
    
    # Default: show overview
    print("🔍 Use --help to see available options")
    print("\nQuick commands:")
    print("  --tables              # List all tables")
    print("  --summary             # Show data summary")
    print("  --table market_data   # View market data")
    print("  --table sentiment_data --symbol AAPL  # View AAPL sentiment")
    print()
    
    # Show tables by default
    list_tables(db_path)

if __name__ == "__main__":
    main()