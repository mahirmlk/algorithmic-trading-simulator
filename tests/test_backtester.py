import unittest

import pandas as pd

from backtester import Backtester


class StaticSignalStrategy:
    def __init__(self, signals, name="StaticSignalStrategy"):
        self.name = name
        self._signals = signals
        self.data = None

    def set_data(self, data):
        self.data = data.copy()

    def generate_signals(self):
        df = self.data.copy()
        df["signal"] = self._signals
        return df


def build_ohlc():
    dates = pd.date_range("2024-01-01", periods=6, freq="D")
    return pd.DataFrame(
        {
            "Date": dates,
            "Open": [100.0, 101.0, 99.0, 98.0, 96.0, 97.0],
            "High": [101.0, 102.0, 100.0, 99.0, 97.0, 98.0],
            "Low": [99.0, 100.0, 98.0, 97.0, 95.0, 96.0],
            "Close": [100.5, 100.0, 98.5, 97.5, 96.5, 97.0],
        }
    )


class BacktesterExecutionTests(unittest.TestCase):
    def test_signal_executes_on_next_open(self):
        df = build_ohlc().iloc[:4].copy()
        strategy = StaticSignalStrategy([1, 0, 0, 0])
        backtester = Backtester(
            initial_capital=100000,
            commission=0.0,
            slippage=0.0,
            execution_timing="next_open",
        )

        results = backtester.run(df, strategy, verbose=False)
        trades = results["trade_log"]

        self.assertEqual(len(trades), 1)
        self.assertEqual(trades.iloc[0]["Action"], "buy")
        self.assertEqual(trades.iloc[0]["Date"], df.iloc[1]["Date"])
        self.assertEqual(trades.iloc[0]["Price"], df.iloc[1]["Open"])

    def test_short_positions_supported_when_enabled(self):
        df = build_ohlc()
        strategy = StaticSignalStrategy([0, -1, 0, 1, 0, 0])
        backtester = Backtester(
            initial_capital=100000,
            commission=0.0,
            slippage=0.0,
            allow_short=True,
            execution_timing="next_open",
        )

        results = backtester.run(df, strategy, verbose=False)
        trades = results["trade_log"]

        self.assertEqual(trades["Action"].tolist(), ["short", "cover"])
        self.assertEqual(results["completed_trades"], 1)
        self.assertGreater(trades.iloc[1]["PnL"], 0)

    def test_short_positions_ignored_when_disabled(self):
        df = build_ohlc()
        strategy = StaticSignalStrategy([0, -1, 0, -1, 0, 0])
        backtester = Backtester(
            initial_capital=100000,
            commission=0.0,
            slippage=0.0,
            allow_short=False,
            execution_timing="next_open",
        )

        results = backtester.run(df, strategy, verbose=False)
        trades = results["trade_log"]
        self.assertTrue(trades.empty)

    def test_next_open_requires_open_column(self):
        df = build_ohlc().drop(columns=["Open"])
        strategy = StaticSignalStrategy([0, 0, 0, 0, 0, 0])
        backtester = Backtester(execution_timing="next_open")

        with self.assertRaises(ValueError):
            backtester.run(df, strategy, verbose=False)


if __name__ == "__main__":
    unittest.main()
