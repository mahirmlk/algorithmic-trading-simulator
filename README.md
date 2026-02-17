# Algorithmic Trading Simulator
# Backtesting Engine built from scratch using NumPy/Pandas

A professional-grade backtesting engine for testing trading strategies against historical data.

## Installation

```bash
pip install numpy pandas matplotlib
```

## Quick Start

```bash
python demo.py
```

## Backtest Execution Model

- Signals are generated on bar `t` and executed on bar `t+1` (default: next bar `Open`) to avoid look-ahead bias.
- Demo data is loaded via `DataHandler` so OHLC columns are normalized and validated before running strategies.
- Engine supports long trades by default and optional short selling (`allow_short=True`) in `Backtester`.

## Project Structure

- `data.py` - Data loading and preprocessing
- `strategies.py` - Trading strategy framework
- `backtester.py` - Core backtesting engine
- `analytics.py` - Performance metrics and reporting
- `demo.py` - Demo script with sample strategies
