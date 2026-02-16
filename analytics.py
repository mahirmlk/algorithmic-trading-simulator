import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from typing import Dict, Optional, List, Tuple
from pathlib import Path


class Analytics:
    """
    Analytics module for calculating performance metrics and generating visualizations.
    """
    
    def __init__(self, results: Optional[Dict] = None):
        """
        Initialize analytics.
        
        Args:
            results: Results dictionary from Backtester
        """
        self.results = results
    
    def set_results(self, results: Dict) -> None:
        """Set results dictionary."""
        self.results = results
    
    def calculate_metrics(self) -> Dict:
        """
        Calculate comprehensive performance metrics.
        
        Returns:
            Dictionary of performance metrics
        """
        if not self.results:
            return {}
        
        equity = self.results.get('equity_curve', pd.DataFrame())
        if equity.empty:
            return {}
        
        equity_values = equity['Equity'].values
        
        # Basic metrics
        metrics = {
            'Initial Capital': self.results.get('initial_capital', 0),
            'Final Equity': equity_values[-1],
            'Total Return': self.results.get('total_return_pct', 0),
            'Total Trades': self.results.get('total_trades', 0),
            'Completed Trades': self.results.get('completed_trades', 0),
        }
        
        # Risk metrics
        metrics.update(self._calculate_risk_metrics(equity_values))
        
        # Trade statistics
        metrics.update(self._calculate_trade_stats())
        
        return metrics
    
    def _calculate_risk_metrics(self, equity: np.ndarray) -> Dict:
        """Calculate risk-related metrics."""
        # Returns
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        metrics = {}
        
        if len(returns) > 0:
            # Annualized metrics (assuming 252 trading days)
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            
            metrics['Avg Daily Return'] = avg_return * 100
            metrics['Daily Volatility'] = std_return * 100
            metrics['Annual Return'] = avg_return * 252 * 100
            metrics['Annual Volatility'] = std_return * np.sqrt(252) * 100
            
            # Sharpe Ratio (assuming risk-free rate = 0)
            metrics['Sharpe Ratio'] = (avg_return / std_return * np.sqrt(252)) if std_return > 0 else 0
            
            # Sortino Ratio (downside deviation)
            downside_returns = returns[returns < 0]
            downside_std = np.std(downside_returns) if len(downside_returns) > 0 else 0
            metrics['Sortino Ratio'] = (avg_return / downside_std * np.sqrt(252)) if downside_std > 0 else 0
            
            # Calmar Ratio (return / max drawdown)
            max_dd = self.results.get('max_drawdown', 0)
            metrics['Calmar Ratio'] = (avg_return * 252 / abs(max_dd)) if max_dd != 0 else 0
        else:
            metrics['Avg Daily Return'] = 0
            metrics['Daily Volatility'] = 0
            metrics['Annual Return'] = 0
            metrics['Annual Volatility'] = 0
            metrics['Sharpe Ratio'] = 0
            metrics['Sortino Ratio'] = 0
            metrics['Calmar Ratio'] = 0
        
        # Drawdown metrics
        metrics['Max Drawdown'] = self.results.get('max_drawdown_pct', 0)
        
        # Calculate average drawdown duration
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        
        in_drawdown = drawdowns < -0.01  # > 1% drawdown
        drawdown_durations = []
        current_duration = 0
        
        for is_dd in in_drawdown:
            if is_dd:
                current_duration += 1
            elif current_duration > 0:
                drawdown_durations.append(current_duration)
                current_duration = 0
        
        if current_duration > 0:
            drawdown_durations.append(current_duration)
        
        metrics['Avg Drawdown Duration'] = np.mean(drawdown_durations) if drawdown_durations else 0
        metrics['Max Drawdown Duration'] = max(drawdown_durations) if drawdown_durations else 0
        
        return metrics
    
    def _calculate_trade_stats(self) -> Dict:
        """Calculate trade statistics."""
        trade_log = self.results.get('trade_log', pd.DataFrame())
        
        if trade_log.empty:
            return {
                'Win Rate': 0,
                'Total Wins': 0,
                'Total Losses': 0,
                'Avg Win': 0,
                'Avg Loss': 0,
                'Largest Win': 0,
                'Largest Loss': 0,
                'Profit Factor': 0,
                'Avg Trade Duration': 0,
            }
        
        # Filter sell trades (completed trades)
        sells = trade_log[trade_log['Action'] == 'sell']
        
        if sells.empty:
            return {
                'Win Rate': 0,
                'Total Wins': 0,
                'Total Losses': 0,
                'Avg Win': 0,
                'Avg Loss': 0,
                'Largest Win': 0,
                'Largest Loss': 0,
                'Profit Factor': 0,
                'Avg Trade Duration': 0,
            }
        
        wins = sells[sells['PnL'] > 0]
        losses = sells[sells['PnL'] <= 0]
        
        stats = {
            'Win Rate': len(wins) / len(sells) * 100 if len(sells) > 0 else 0,
            'Total Wins': len(wins),
            'Total Losses': len(losses),
            'Avg Win': wins['PnL'].mean() if len(wins) > 0 else 0,
            'Avg Loss': losses['PnL'].mean() if len(losses) > 0 else 0,
            'Largest Win': wins['PnL'].max() if len(wins) > 0 else 0,
            'Largest Loss': losses['PnL'].min() if len(losses) > 0 else 0,
            'Profit Factor': self.results.get('profit_factor', 0),
        }
        
        # Calculate trade duration
        buys = trade_log[trade_log['Action'] == 'buy']
        if not buys.empty and not sells.empty:
            # Match buys with sells (simple approach - sequential)
            sells_sorted = sells.sort_values('Date')
            buys_sorted = buys.sort_values('Date')
            
            durations = []
            buy_idx = 0
            for _, sell in sells_sorted.iterrows():
                if buy_idx < len(buys_sorted):
                    buy_date = buys_sorted.iloc[buy_idx]['Date']
                    sell_date = sell['Date']
                    duration = (sell_date - buy_date).days
                    durations.append(duration)
                    buy_idx += 1
            
            stats['Avg Trade Duration'] = np.mean(durations) if durations else 0
        else:
            stats['Avg Trade Duration'] = 0
        
        return stats
    
    def plot_equity_curve(
        self, 
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot equity curve.
        
        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.results:
            print("No results to plot")
            return None
        
        equity = self.results.get('equity_curve', pd.DataFrame())
        if equity.empty:
            print("No equity data to plot")
            return None
        
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot equity curve
        ax.plot(equity['Date'], equity['Equity'], linewidth=2, color='#2ecc71')
        
        # Fill under curve
        ax.fill_between(
            equity['Date'], 
            equity['Equity'], 
            alpha=0.3, 
            color='#2ecc71'
        )
        
        # Plot initial capital line
        initial = self.results.get('initial_capital', 0)
        ax.axhline(y=initial, color='#e74c3c', linestyle='--', linewidth=1, alpha=0.7)
        
        # Formatting
        ax.set_title('Equity Curve', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format y-axis as currency
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        # Add return annotation
        total_return = self.results.get('total_return_pct', 0)
        final_equity = equity['Equity'].iloc[-1]
        
        textstr = f'Final: ${final_equity:,.2f}\nReturn: {total_return:.2f}%'
        props = dict(boxstyle='round', facecolor='white', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_drawdown(
        self, 
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot drawdown chart.
        
        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.results:
            return None
        
        equity = self.results.get('equity_curve', pd.DataFrame())
        if equity.empty:
            return None
        
        # Calculate drawdowns
        equity_values = equity['Equity'].values
        running_max = np.maximum.accumulate(equity_values)
        drawdowns = (equity_values - running_max) / running_max * 100
        
        fig, ax = plt.subplots(figsize=(12, 4))
        
        # Plot drawdowns
        ax.fill_between(equity['Date'], drawdowns, 0, 
                        color='#e74c3c', alpha=0.5)
        ax.plot(equity['Date'], drawdowns, color='#e74c3c', linewidth=1)
        
        # Formatting
        ax.set_title('Drawdown', fontsize=14, fontweight='bold')
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Drawdown (%)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def plot_trade_pnl(
        self, 
        save_path: Optional[str] = None,
        show: bool = True
    ) -> plt.Figure:
        """
        Plot trade P&L distribution.
        
        Args:
            save_path: Path to save the plot
            show: Whether to display the plot
            
        Returns:
            Matplotlib figure
        """
        if not self.results:
            return None
        
        trade_log = self.results.get('trade_log', pd.DataFrame())
        if trade_log.empty:
            return None
        
        sells = trade_log[trade_log['Action'] == 'sell']
        if sells.empty:
            return None
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram of P&L
        ax1.hist(sells['PnL'], bins=20, edgecolor='black', alpha=0.7, color='#3498db')
        ax1.axvline(x=0, color='black', linestyle='-', linewidth=1)
        ax1.set_title('Trade P&L Distribution', fontsize=14, fontweight='bold')
        ax1.set_xlabel('P&L ($)', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # Cumulative P&L
        sells_sorted = sells.sort_values('Date')
        cumulative_pnl = sells_sorted['PnL'].cumsum()
        
        ax2.plot(sells_sorted['Date'], cumulative_pnl, linewidth=2, color='#3498db')
        ax2.fill_between(sells_sorted['Date'], cumulative_pnl, 0, 
                        alpha=0.3, color='#3498db')
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=1)
        ax2.set_title('Cumulative P&L', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Cumulative P&L ($)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if show:
            plt.show()
        
        return fig
    
    def generate_report(
        self, 
        output_dir: str = 'reports',
        strategy_name: str = 'strategy'
    ) -> None:
        """
        Generate comprehensive report with all visualizations.
        
        Args:
            output_dir: Directory to save reports
            strategy_name: Name for file naming
        """
        if not self.results:
            print("No results to generate report")
            return
        
        # Create output directory
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Generate all plots
        self.plot_equity_curve(
            save_path=f"{output_dir}/{strategy_name}_equity.png",
            show=False
        )
        
        self.plot_drawdown(
            save_path=f"{output_dir}/{strategy_name}_drawdown.png",
            show=False
        )
        
        self.plot_trade_pnl(
            save_path=f"{output_dir}/{strategy_name}_pnl.png",
            show=False
        )
        
        # Save metrics to CSV
        metrics = self.calculate_metrics()
        metrics_df = pd.DataFrame([metrics]).T
        metrics_df.columns = ['Value']
        metrics_df.to_csv(f"{output_dir}/{strategy_name}_metrics.csv")
        
        # Save trade log
        trade_log = self.results.get('trade_log', pd.DataFrame())
        if not trade_log.empty:
            trade_log.to_csv(f"{output_dir}/{strategy_name}_trades.csv", index=False)
        
        # Save equity curve
        equity = self.results.get('equity_curve', pd.DataFrame())
        if not equity.empty:
            equity.to_csv(f"{output_dir}/{strategy_name}_equity.csv", index=False)
        
        print(f"\nReport generated in '{output_dir}/'")
        print(f"  - {strategy_name}_equity.png")
        print(f"  - {strategy_name}_drawdown.png")
        print(f"  - {strategy_name}_pnl.png")
        print(f"  - {strategy_name}_metrics.csv")
        print(f"  - {strategy_name}_trades.csv")
        print(f"  - {strategy_name}_equity.csv")
    
    def print_metrics(self) -> None:
        """Print formatted metrics."""
        metrics = self.calculate_metrics()
        
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print("="*60)
        
        # Group metrics
        print("\nRETURN METRICS")
        print("-"*40)
        print(f"  Initial Capital:      ${metrics.get('Initial Capital', 0):>12,.2f}")
        print(f"  Final Equity:         ${metrics.get('Final Equity', 0):>12,.2f}")
        print(f"  Total Return:         {metrics.get('Total Return', 0):>12.2f}%")
        print(f"  Annual Return:        {metrics.get('Annual Return', 0):>12.2f}%")
        
        print("\nRISK METRICS")
        print("-"*40)
        print(f"  Sharpe Ratio:         {metrics.get('Sharpe Ratio', 0):>12.2f}")
        print(f"  Sortino Ratio:        {metrics.get('Sortino Ratio', 0):>12.2f}")
        print(f"  Calmar Ratio:         {metrics.get('Calmar Ratio', 0):>12.2f}")
        print(f"  Annual Volatility:    {metrics.get('Annual Volatility', 0):>12.2f}%")
        print(f"  Max Drawdown:         {metrics.get('Max Drawdown', 0):>12.2f}%")
        
        print("\nTRADE STATISTICS")
        print("-"*40)
        print(f"  Total Trades:        {metrics.get('Total Trades', 0):>12}")
        print(f"  Win Rate:             {metrics.get('Win Rate', 0):>12.1f}%")
        print(f"  Profit Factor:        {metrics.get('Profit Factor', 0):>12.2f}")
        print(f"  Avg Win:              ${metrics.get('Avg Win', 0):>12,.2f}")
        print(f"  Avg Loss:             ${metrics.get('Avg Loss', 0):>12,.2f}")
        
        print("="*60)
