import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: pd.Timestamp
    action: str  # 'buy' or 'sell'
    price: float
    quantity: float
    commission: float = 0.0
    pnl: float = 0.0  # Profit/Loss (for sells)
    pnl_pct: float = 0.0  # P&L as percentage


@dataclass
class Position:
    """Represents an open position."""
    entry_timestamp: pd.Timestamp
    entry_price: float
    quantity: float
    direction: str = 'long'  # 'long' or 'short'


@dataclass
class Portfolio:
    """Represents the current portfolio state."""
    cash: float
    positions: List[Position] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[pd.Timestamp] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)
    
    @property
    def total_value(self, current_price: float = 0) -> float:
        """Calculate total portfolio value."""
        position_value = sum(p.quantity * p.entry_price for p in self.positions)
        return self.cash + position_value
    
    @property
    def position_count(self) -> int:
        """Get number of open positions."""
        return len(self.positions)


class Backtester:
    """
    Core backtesting engine.
    
    Executes trading strategies against historical data and tracks portfolio performance.
    """
    
    def __init__(
        self,
        initial_capital: float = 100000.0,
        commission: float = 0.001,  # 0.1% commission
        slippage: float = 0.0005,  # 0.05% slippage
        position_sizing: str = 'fixed',  # 'fixed' or 'percent'
        max_position_pct: float = 1.0,  # Max % of portfolio per position
    ):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade (as decimal)
            slippage: Slippage rate per trade (as decimal)
            position_sizing: 'fixed' (fixed $ amount) or 'percent' (% of portfolio)
            max_position_pct: Maximum position size as % of portfolio
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.max_position_pct = max_position_pct
        
        # State
        self.portfolio = Portfolio(cash=initial_capital)
        self.data: Optional[pd.DataFrame] = None
        self.signals: Optional[pd.DataFrame] = None
        self.results: Optional[Dict] = None
    
    def run(
        self, 
        data: pd.DataFrame, 
        strategy,
        verbose: bool = True
    ) -> Dict:
        """
        Run a backtest.
        
        Args:
            data: Historical OHLCV data
            strategy: A strategy instance (must have generate_signals method)
            verbose: Print progress messages
            
        Returns:
            Dictionary with backtest results
        """
        # Reset state
        self.portfolio = Portfolio(cash=self.initial_capital)
        
        # Prepare data
        self.data = data.copy()
        
        # Generate signals
        if verbose:
            print(f"Running backtest with strategy: {strategy.name}")
        
        strategy.set_data(self.data)
        self.signals = strategy.generate_signals()
        
        # Execute trades
        self._execute_signals(verbose)
        
        # Calculate results
        self.results = self._calculate_results()
        
        if verbose:
            self._print_summary()
        
        return self.results
    
    def _execute_signals(self, verbose: bool) -> None:
        """Execute trading signals."""
        # Get signal column
        if 'signal' not in self.signals.columns:
            raise ValueError("Strategy must generate 'signal' column")
        
        # Track equity curve
        current_price = self.data.iloc[0]['Close']
        self.portfolio.equity_curve.append(
            self.portfolio.cash + (self.portfolio.position_count * current_price)
        )
        self.portfolio.timestamps.append(self.data.iloc[0]['Date'])
        
        # Iterate through signals
        for i in range(len(self.signals)):
            row = self.signals.iloc[i]
            signal = row['signal']
            timestamp = row['Date']
            price = row['Close']
            
            # Skip if no signal
            if signal == 0:
                # Still track equity
                self.portfolio.equity_curve.append(
                    self._get_portfolio_value(price)
                )
                self.portfolio.timestamps.append(timestamp)
                continue
            
            # Apply slippage
            exec_price = price * (1 + self.slippage) if signal > 0 else price * (1 - self.slippage)
            
            if signal > 0:  # Buy signal
                self._execute_buy(timestamp, exec_price)
            elif signal < 0:  # Sell signal
                self._execute_sell(timestamp, exec_price)
            
            # Track equity
            self.portfolio.equity_curve.append(
                self._get_portfolio_value(exec_price)
            )
            self.portfolio.timestamps.append(timestamp)
    
    def _execute_buy(self, timestamp: pd.Timestamp, price: float) -> None:
        """Execute a buy order."""
        # Calculate position size
        if self.position_sizing == 'percent':
            available_capital = self.portfolio.cash * self.max_position_pct
        else:
            available_capital = min(self.portfolio.cash, self.initial_capital * 0.1)
        
        # Calculate quantity (leave some buffer for commission)
        quantity = available_capital / price
        commission_cost = available_capital * self.commission
        
        # Check if we have enough cash
        total_cost = (quantity * price) + commission_cost
        if total_cost > self.portfolio.cash:
            quantity = (self.portfolio.cash - commission_cost) / price
        
        if quantity <= 0:
            return
        
        # Execute trade
        commission_cost = quantity * price * self.commission
        
        self.portfolio.cash -= (quantity * price + commission_cost)
        
        position = Position(
            entry_timestamp=timestamp,
            entry_price=price,
            quantity=quantity,
            direction='long'
        )
        self.portfolio.positions.append(position)
        
        trade = Trade(
            timestamp=timestamp,
            action='buy',
            price=price,
            quantity=quantity,
            commission=commission_cost
        )
        self.portfolio.trades.append(trade)
    
    def _execute_sell(self, timestamp: pd.Timestamp, price: float) -> None:
        """Execute a sell order (close position)."""
        if not self.portfolio.positions:
            return
        
        # Close all positions
        for position in self.portfolio.positions:
            position_value = position.quantity * price
            commission_cost = position_value * self.commission
            pnl = position_value - (position.quantity * position.entry_price) - commission_cost
            pnl_pct = (price - position.entry_price) / position.entry_price
            
            self.portfolio.cash += (position_value - commission_cost)
            
            trade = Trade(
                timestamp=timestamp,
                action='sell',
                price=price,
                quantity=position.quantity,
                commission=commission_cost,
                pnl=pnl,
                pnl_pct=pnl_pct
            )
            self.portfolio.trades.append(trade)
        
        # Clear positions
        self.portfolio.positions.clear()
    
    def _get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        position_value = sum(
            p.quantity * current_price for p in self.portfolio.positions
        )
        return self.portfolio.cash + position_value
    
    def _calculate_results(self) -> Dict:
        """Calculate performance metrics."""
        equity = np.array(self.portfolio.equity_curve)
        timestamps = self.portfolio.timestamps
        trades = self.portfolio.trades
        
        if len(equity) == 0:
            return {}
        
        # Basic metrics
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        total_trades = len([t for t in trades if t.action == 'buy'])
        completed_trades = len([t for t in trades if t.action == 'sell'])
        
        # Calculate returns series
        returns = np.diff(equity) / equity[:-1]
        returns = returns[~np.isnan(returns) & ~np.isinf(returns)]
        
        # Risk metrics
        if len(returns) > 0:
            avg_return = np.mean(returns)
            std_return = np.std(returns)
            sharpe_ratio = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
        else:
            avg_return = std_return = sharpe_ratio = 0
        
        # Drawdown
        running_max = np.maximum.accumulate(equity)
        drawdowns = (equity - running_max) / running_max
        max_drawdown = np.min(drawdowns) if len(drawdowns) > 0 else 0
        
        # Win rate
        sell_trades = [t for t in trades if t.action == 'sell']
        if sell_trades:
            winning_trades = sum(1 for t in sell_trades if t.pnl > 0)
            win_rate = winning_trades / len(sell_trades)
            avg_win = np.mean([t.pnl for t in sell_trades if t.pnl > 0]) if winning_trades > 0 else 0
            losing_trades = [t for t in sell_trades if t.pnl <= 0]
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            profit_factor = abs(avg_win / avg_loss) if avg_loss != 0 else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = 0
        
        # Trade log
        trade_log = pd.DataFrame([
            {
                'Date': t.timestamp,
                'Action': t.action,
                'Price': t.price,
                'Quantity': t.quantity,
                'Value': t.price * t.quantity,
                'Commission': t.commission,
                'PnL': t.pnl,
                'PnL_Pct': t.pnl_pct * 100
            }
            for t in trades
        ])
        
        # Equity curve DataFrame
        equity_curve = pd.DataFrame({
            'Date': timestamps,
            'Equity': equity
        })
        
        results = {
            'initial_capital': self.initial_capital,
            'final_equity': equity[-1],
            'total_return': total_return,
            'total_return_pct': total_return * 100,
            'total_trades': total_trades,
            'completed_trades': completed_trades,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown * 100,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trade_log': trade_log,
            'equity_curve': equity_curve,
        }
        
        return results
    
    def _print_summary(self) -> None:
        """Print backtest summary."""
        if not self.results:
            return
        
        r = self.results
        print("\n" + "="*50)
        print("BACKTEST SUMMARY")
        print("="*50)
        print(f"Initial Capital:     ${r['initial_capital']:,.2f}")
        print(f"Final Equity:       ${r['final_equity']:,.2f}")
        print(f"Total Return:       {r['total_return_pct']:.2f}%")
        print(f"Total Trades:       {r['total_trades']}")
        print("-"*50)
        print(f"Sharpe Ratio:        {r['sharpe_ratio']:.2f}")
        print(f"Max Drawdown:       {r['max_drawdown_pct']:.2f}%")
        print(f"Win Rate:           {r['win_rate']*100:.1f}%")
        print(f"Profit Factor:      {r['profit_factor']:.2f}")
        print("="*50)
    
    def get_results(self) -> Optional[Dict]:
        """Get backtest results."""
        return self.results
    
    def get_trades(self) -> List[Trade]:
        """Get all trades."""
        return self.portfolio.trades
    
    def get_equity_curve(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if self.results:
            return self.results['equity_curve']
        return pd.DataFrame()
    
    def get_trade_log(self) -> pd.DataFrame:
        """Get trade log as DataFrame."""
        if self.results:
            return self.results['trade_log']
        return pd.DataFrame()
