import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for headless environments

from data import DataHandler
from strategies import SMACrossover, RSIMeanReversion, BollingerBandsStrategy, MomentumStrategy
from backtester import Backtester
from analytics import Analytics


def generate_sample_data():
    """Generate synthetic OHLCV data for demonstration."""
    print("Generating sample data...")
    
    # Generate 2 years of daily data
    n_days = 504  # ~2 years of trading days
    
    # Generate random walk with drift
    np.random.seed(42)
    drift = 0.0002  # Small upward drift
    volatility = 0.02  # 2% daily volatility
    
    returns = np.random.normal(drift, volatility, n_days)
    close_prices = 100 * np.exp(np.cumsum(returns))
    
    # Generate OHLC from close
    data = []
    base_date = pd.Timestamp('2022-01-01')
    
    for i, close in enumerate(close_prices):
        # Add realistic OHLC variation
        daily_range = close * np.random.uniform(0.005, 0.03)
        
        high = close + np.random.uniform(0, daily_range)
        low = close - np.random.uniform(0, daily_range)
        open_price = close + np.random.uniform(-daily_range/2, daily_range/2)
        
        # Ensure OHLC integrity
        high = max(high, close, open_price)
        low = min(low, close, open_price)
        
        # Generate volume with some patterns
        base_volume = 1_000_000
        volume = int(base_volume * np.random.uniform(0.5, 2.5))
        
        # Skip weekends
        date = base_date + pd.Timedelta(days=i)
        if date.dayofweek >= 5:
            continue
        
        data.append({
            'Date': date,
            'Open': round(open_price, 2),
            'High': round(high, 2),
            'Low': round(low, 2),
            'Close': round(close, 2),
            'Volume': volume
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv('sample_data.csv', index=False)
    print(f"Generated {len(df)} days of sample data")
    
    return df


def load_or_generate_validated_data(filepath: str = 'sample_data.csv') -> pd.DataFrame:
    """Load sample data through DataHandler validation."""
    handler = DataHandler()
    try:
        df = handler.load_csv(filepath)
        print("Loaded existing sample data (validated)")
    except FileNotFoundError:
        generate_sample_data()
        df = handler.load_csv(filepath)
        print("Generated sample data and validated with DataHandler")

    return df


def run_strategy_demo():
    """Run demonstration with different strategies."""
    
    # Load or generate data via DataHandler to enforce schema checks.
    df = load_or_generate_validated_data('sample_data.csv')
    
    print(f"\nData range: {df['Date'].min().date()} to {df['Date'].max().date()}")
    print(f"Data points: {len(df)}")
    print(f"Price range: ${df['Close'].min():.2f} - ${df['Close'].max():.2f}")
    
    # Initialize backtester
    initial_capital = 100000
    commission = 0.001  # 0.1%
    
    backtester = Backtester(
        initial_capital=initial_capital,
        commission=commission,
        slippage=0.0005,
        position_sizing='percent',
        max_position_pct=0.95,
        allow_short=False,
        execution_timing='next_open'
    )
    
    # Run different strategies
    strategies = [
        SMACrossover(short_period=20, long_period=50),
        RSIMeanReversion(period=14, oversold=30, overbought=70),
        BollingerBandsStrategy(period=20, num_std=2.0),
        MomentumStrategy(period=20, threshold=0.05),
    ]
    
    results_summary = []
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Running: {strategy.name}")
        print('='*60)
        
        # Run backtest
        results = backtester.run(df, strategy, verbose=True)
        
        # Get analytics
        analytics = Analytics(results)
        metrics = analytics.calculate_metrics()
        
        # Store summary
        results_summary.append({
            'Strategy': strategy.name,
            'Return': results['total_return_pct'],
            'Sharpe': results['sharpe_ratio'],
            'Max DD': results['max_drawdown_pct'],
            'Win Rate': results['win_rate'] * 100,
            'Trades': results['completed_trades'],
        })
        
        # Generate reports
        analytics.generate_report(
            output_dir='reports',
            strategy_name=strategy.name.replace(' ', '_').lower()
        )
        
        # Print detailed metrics
        analytics.print_metrics()
    
    # Print comparison summary
    print("\n" + "="*60)
    print("STRATEGY COMPARISON")
    print("="*60)
    
    summary_df = pd.DataFrame(results_summary)
    print(summary_df.to_string(index=False))
    
    # Save comparison
    summary_df.to_csv('strategy_comparison.csv', index=False)
    print("\nResults saved to 'reports/' directory")


def run_single_strategy_demo():
    """Run a single strategy demo with detailed output."""
    
    # Generate then reload via DataHandler for consistent validation.
    generate_sample_data()
    handler = DataHandler()
    df = handler.load_csv('sample_data.csv')
    
    # Create strategy
    strategy = SMACrossover(short_period=20, long_period=50)
    
    # Run backtest
    backtester = Backtester(
        initial_capital=100000,
        commission=0.001,
        slippage=0.0005,
        allow_short=False,
        execution_timing='next_open'
    )
    
    results = backtester.run(df, strategy, verbose=True)
    
    # Analyze results
    analytics = Analytics(results)
    analytics.print_metrics()
    
    # Generate plots
    analytics.plot_equity_curve(save_path='reports/equity_curve.png', show=False)
    analytics.plot_drawdown(save_path='reports/drawdown.png', show=False)
    analytics.plot_trade_pnl(save_path='reports/trade_pnl.png', show=False)
    
    # Show trade log
    trades = results['trade_log']
    if not trades.empty:
        print("\nTRADE LOG (first 10 trades):")
        print(trades.head(10).to_string(index=False))
    
    print("\nDemo complete. Check the 'reports/' folder for visualizations.")


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--quick':
        # Quick single strategy demo
        run_single_strategy_demo()
    else:
        # Full demo with multiple strategies
        run_strategy_demo()
