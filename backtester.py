import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from dataclasses import dataclass, field


@dataclass
class Trade:
    """Represents a single trade."""
    timestamp: pd.Timestamp
    action: str  # 'buy', 'sell', 'short', 'cover'
    price: float
    quantity: float
    commission: float = 0.0
    pnl: float = 0.0  # Profit/Loss (for closing trades)
    pnl_pct: float = 0.0  # P&L as percentage


@dataclass
class Position:
    """Represents an open position."""
    entry_timestamp: pd.Timestamp
    entry_price: float
    quantity: float
    direction: str = 'long'  # 'long' or 'short'
    entry_commission: float = 0.0


@dataclass
class Portfolio:
    """Represents the current portfolio state."""
    cash: float
    positions: List[Position] = field(default_factory=list)
    equity_curve: List[float] = field(default_factory=list)
    timestamps: List[pd.Timestamp] = field(default_factory=list)
    trades: List[Trade] = field(default_factory=list)

    def total_value(self, current_price: float = 0.0) -> float:
        """Calculate total portfolio value."""
        long_value = sum(
            p.quantity * current_price for p in self.positions if p.direction == 'long'
        )
        short_liability = sum(
            p.quantity * current_price for p in self.positions if p.direction == 'short'
        )
        return self.cash + long_value - short_liability
    
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
        allow_short: bool = False,
        execution_timing: str = 'next_open',  # 'next_open' or 'next_close'
    ):
        """
        Initialize the backtester.
        
        Args:
            initial_capital: Starting capital
            commission: Commission rate per trade (as decimal)
            slippage: Slippage rate per trade (as decimal)
            position_sizing: 'fixed' (fixed $ amount) or 'percent' (% of portfolio)
            max_position_pct: Maximum position size as % of portfolio
            allow_short: Enable short selling when signal == -1 and no long exists
            execution_timing: Order fill timing ('next_open' preferred to avoid look-ahead bias)
        """
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_sizing = position_sizing
        self.max_position_pct = max_position_pct
        self.allow_short = allow_short
        self.execution_timing = execution_timing
        
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
        required_cols = {'Date', 'Close'}
        if self.execution_timing == 'next_open':
            required_cols.add('Open')

        missing = required_cols - set(self.data.columns)
        if missing:
            raise ValueError(f"Missing required columns for backtest: {missing}")
        
        self.data = self.data.sort_values('Date').reset_index(drop=True)
        
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
        """Execute trading signals on the next bar to avoid look-ahead bias."""
        # Get signal column
        if 'signal' not in self.signals.columns:
            raise ValueError("Strategy must generate 'signal' column")

        # Track mark-to-market equity at each bar close.
        for i in range(len(self.signals)):
            row = self.signals.iloc[i]
            timestamp = row['Date']
            close_price = row['Close']

            self.portfolio.equity_curve.append(self._get_portfolio_value(close_price))
            self.portfolio.timestamps.append(timestamp)

            signal = row['signal']
            if pd.isna(signal) or signal == 0:
                continue

            # No future bar to execute on.
            if i >= len(self.signals) - 1:
                continue

            next_row = self.signals.iloc[i + 1]
            exec_timestamp = next_row['Date']
            raw_price = next_row['Open'] if self.execution_timing == 'next_open' else next_row['Close']

            if pd.isna(raw_price) or raw_price <= 0:
                continue

            exec_price = self._apply_slippage(raw_price, signal)
            self._execute_signal_action(signal, exec_timestamp, exec_price)

    def _apply_slippage(self, price: float, signal: float) -> float:
        """Apply slippage to execution price."""
        return price * (1 + self.slippage) if signal > 0 else price * (1 - self.slippage)

    def _execute_signal_action(self, signal: float, timestamp: pd.Timestamp, price: float) -> None:
        """Route a signal to the correct order type."""
        if signal > 0:
            if self._has_open_positions('short'):
                self._execute_cover(timestamp, price)
            elif not self._has_open_positions('long'):
                self._execute_buy(timestamp, price)
        elif signal < 0:
            if self._has_open_positions('long'):
                self._execute_sell(timestamp, price)
            elif self.allow_short and not self._has_open_positions('short'):
                self._execute_short(timestamp, price)

    def _has_open_positions(self, direction: str) -> bool:
        """Return True if an open position exists for a direction."""
        return any(p.direction == direction for p in self.portfolio.positions)

    def _get_position_budget(self) -> float:
        """Calculate capital budget for a new position."""
        if self.position_sizing == 'percent':
            return max(0.0, self.portfolio.cash * self.max_position_pct)
        return max(0.0, min(self.portfolio.cash, self.initial_capital * 0.1))

    def _get_order_quantity(self, price: float) -> float:
        """Calculate quantity while reserving commission."""
        if price <= 0:
            return 0.0
        budget = self._get_position_budget()
        if budget <= 0:
            return 0.0
        return budget / (price * (1 + self.commission))
    
    def _execute_buy(self, timestamp: pd.Timestamp, price: float) -> None:
        """Execute a buy order."""
        if self._has_open_positions('long'):
            return

        quantity = self._get_order_quantity(price)
        if quantity <= 0:
            return

        notional = quantity * price
        commission_cost = notional * self.commission
        total_cost = notional + commission_cost
        if total_cost > self.portfolio.cash:
            quantity = self.portfolio.cash / (price * (1 + self.commission))
            notional = quantity * price
            commission_cost = notional * self.commission
            total_cost = notional + commission_cost

        if quantity <= 0:
            return

        self.portfolio.cash -= total_cost
        
        position = Position(
            entry_timestamp=timestamp,
            entry_price=price,
            quantity=quantity,
            direction='long',
            entry_commission=commission_cost
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
    
    def _execute_short(self, timestamp: pd.Timestamp, price: float) -> None:
        """Execute a short order."""
        if self._has_open_positions('short'):
            return

        quantity = self._get_order_quantity(price)
        if quantity <= 0:
            return

        notional = quantity * price
        commission_cost = notional * self.commission

        # Short sale proceeds are credited; commission is paid.
        self.portfolio.cash += (notional - commission_cost)

        position = Position(
            entry_timestamp=timestamp,
            entry_price=price,
            quantity=quantity,
            direction='short',
            entry_commission=commission_cost
        )
        self.portfolio.positions.append(position)

        trade = Trade(
            timestamp=timestamp,
            action='short',
            price=price,
            quantity=quantity,
            commission=commission_cost
        )
        self.portfolio.trades.append(trade)

    def _execute_sell(self, timestamp: pd.Timestamp, price: float) -> None:
        """Close all long positions."""
        self._close_positions(direction='long', timestamp=timestamp, price=price, close_action='sell')

    def _execute_cover(self, timestamp: pd.Timestamp, price: float) -> None:
        """Close all short positions."""
        self._close_positions(direction='short', timestamp=timestamp, price=price, close_action='cover')

    def _close_positions(
        self,
        direction: str,
        timestamp: pd.Timestamp,
        price: float,
        close_action: str
    ) -> None:
        """Close all positions for a given direction."""
        remaining_positions: List[Position] = []

        for position in self.portfolio.positions:
            if position.direction != direction:
                remaining_positions.append(position)
                continue

            notional = position.quantity * price
            commission_cost = notional * self.commission

            if direction == 'long':
                pnl = (
                    notional
                    - (position.quantity * position.entry_price)
                    - position.entry_commission
                    - commission_cost
                )
                pnl_pct = (price - position.entry_price) / position.entry_price
                self.portfolio.cash += (notional - commission_cost)
            else:
                pnl = (
                    (position.quantity * position.entry_price)
                    - notional
                    - position.entry_commission
                    - commission_cost
                )
                pnl_pct = (position.entry_price - price) / position.entry_price
                self.portfolio.cash -= (notional + commission_cost)

            trade = Trade(
                timestamp=timestamp,
                action=close_action,
                price=price,
                quantity=position.quantity,
                commission=commission_cost,
                pnl=pnl,
                pnl_pct=pnl_pct
            )
            self.portfolio.trades.append(trade)

        self.portfolio.positions = remaining_positions
    
    def _get_portfolio_value(self, current_price: float) -> float:
        """Calculate current portfolio value."""
        long_value = sum(
            p.quantity * current_price for p in self.portfolio.positions if p.direction == 'long'
        )
        short_liability = sum(
            p.quantity * current_price for p in self.portfolio.positions if p.direction == 'short'
        )
        return self.portfolio.cash + long_value - short_liability
    
    def _calculate_results(self) -> Dict:
        """Calculate performance metrics."""
        equity = np.array(self.portfolio.equity_curve)
        timestamps = self.portfolio.timestamps
        trades = self.portfolio.trades
        
        if len(equity) == 0:
            return {}
        
        # Basic metrics
        total_return = (equity[-1] - self.initial_capital) / self.initial_capital
        open_actions = {'buy', 'short'}
        close_actions = {'sell', 'cover'}
        total_trades = len([t for t in trades if t.action in open_actions])
        completed_trades = len([t for t in trades if t.action in close_actions])
        
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
        closed_trades = [t for t in trades if t.action in close_actions]
        if closed_trades:
            winning_trades = sum(1 for t in closed_trades if t.pnl > 0)
            win_rate = winning_trades / len(closed_trades)
            avg_win = np.mean([t.pnl for t in closed_trades if t.pnl > 0]) if winning_trades > 0 else 0
            losing_trades = [t for t in closed_trades if t.pnl <= 0]
            avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
            gross_profit = sum(t.pnl for t in closed_trades if t.pnl > 0)
            gross_loss = abs(sum(t.pnl for t in closed_trades if t.pnl < 0))
            if gross_loss > 0:
                profit_factor = gross_profit / gross_loss
            elif gross_profit > 0:
                profit_factor = float('inf')
            else:
                profit_factor = 0
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
            'allow_short': self.allow_short,
            'execution_timing': self.execution_timing,
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
