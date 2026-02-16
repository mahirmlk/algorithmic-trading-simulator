import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass


@dataclass
class Signal:
    """Represents a trading signal."""
    timestamp: pd.Timestamp
    action: str  # 'buy', 'sell', 'hold'
    price: float
    quantity: float = 0.0
    reason: str = ""


class BaseStrategy(ABC):
    """
    Base class for all trading strategies.
    Users should extend this class and implement the generate_signals method.
    """
    
    def __init__(self, name: str = "BaseStrategy"):
        self.name = name
        self.data: Optional[pd.DataFrame] = None
        self.signals: List[Signal] = []
    
    def set_data(self, data: pd.DataFrame) -> None:
        """Set the historical data for the strategy."""
        self.data = data.copy()
    
    @abstractmethod
    def generate_signals(self) -> pd.DataFrame:
        """
        Generate trading signals based on the strategy logic.
        
        Returns:
            DataFrame with 'signal' column:
            - 1 = Buy
            - -1 = Sell
            - 0 = Hold
        """
        pass
    
    def get_signals(self) -> List[Signal]:
        """Get the list of trading signals."""
        return self.signals
    
    def _ensure_data(self) -> None:
        """Ensure data is loaded before generating signals."""
        if self.data is None:
            raise ValueError("Data not set. Call set_data() first.")


class SMACrossover(BaseStrategy):
    """
    Simple Moving Average Crossover Strategy.
    
    Buy when short SMA crosses above long SMA.
    Sell when short SMA crosses below long SMA.
    """
    
    def __init__(self, short_period: int = 20, long_period: int = 50):
        super().__init__(name=f"SMA_Cross_{short_period}_{long_period}")
        self.short_period = short_period
        self.long_period = long_period
    
    def generate_signals(self) -> pd.DataFrame:
        self._ensure_data()
        df = self.data.copy()
        
        # Calculate SMAs
        df['SMA_Short'] = df['Close'].rolling(window=self.short_period).mean()
        df['SMA_Long'] = df['Close'].rolling(window=self.long_period).mean()
        
        # Generate signals
        df['signal'] = 0
        
        # Buy signal: short SMA crosses above long SMA
        df.loc[
            (df['SMA_Short'] > df['SMA_Long']) & 
            (df['SMA_Short'].shift(1) <= df['SMA_Long'].shift(1)),
            'signal'
        ] = 1
        
        # Sell signal: short SMA crosses below long SMA
        df.loc[
            (df['SMA_Short'] < df['SMA_Long']) & 
            (df['SMA_Short'].shift(1) >= df['SMA_Long'].shift(1)),
            'signal'
        ] = -1
        
        return df


class RSIMeanReversion(BaseStrategy):
    """
    RSI Mean Reversion Strategy.
    
    Buy when RSI is oversold (< 30).
    Sell when RSI is overbought (> 70).
    """
    
    def __init__(self, period: int = 14, oversold: float = 30, overbought: float = 70):
        super().__init__(name=f"RSI_{period}")
        self.period = period
        self.oversold = oversold
        self.overbought = overbought
    
    def generate_signals(self) -> pd.DataFrame:
        self._ensure_data()
        df = self.data.copy()
        
        # Calculate RSI
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = (-delta).where(delta < 0, 0)
        
        avg_gain = gain.rolling(window=self.period).mean()
        avg_loss = loss.rolling(window=self.period).mean()
        
        rs = avg_gain / avg_loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Generate signals
        df['signal'] = 0
        
        # Buy when RSI crosses above oversold threshold
        df.loc[
            (df['RSI'] > self.oversold) & 
            (df['RSI'].shift(1) <= self.oversold),
            'signal'
        ] = 1
        
        # Sell when RSI crosses below overbought threshold
        df.loc[
            (df['RSI'] < self.overbought) & 
            (df['RSI'].shift(1) >= self.overbought),
            'signal'
        ] = -1
        
        return df


class BollingerBandsStrategy(BaseStrategy):
    """
    Bollinger Bands Breakout Strategy.
    
    Buy when price crosses below lower band (oversold).
    Sell when price crosses above upper band (overbought).
    """
    
    def __init__(self, period: int = 20, num_std: float = 2.0):
        super().__init__(name=f"Bollinger_{period}_{num_std}")
        self.period = period
        self.num_std = num_std
    
    def generate_signals(self) -> pd.DataFrame:
        self._ensure_data()
        df = self.data.copy()
        
        # Calculate Bollinger Bands
        df['BB_Middle'] = df['Close'].rolling(window=self.period).mean()
        rolling_std = df['Close'].rolling(window=self.period).std()
        df['BB_Upper'] = df['BB_Middle'] + (rolling_std * self.num_std)
        df['BB_Lower'] = df['BB_Middle'] - (rolling_std * self.num_std)
        
        # Calculate %B
        df['BB_Percent'] = (df['Close'] - df['BB_Lower']) / (df['BB_Upper'] - df['BB_Lower'])
        
        # Generate signals
        df['signal'] = 0
        
        # Buy when price crosses below lower band (oversold)
        df.loc[
            (df['Close'] < df['BB_Lower']) & 
            (df['Close'].shift(1) >= df['BB_Lower'].shift(1)),
            'signal'
        ] = 1
        
        # Sell when price crosses above upper band (overbought)
        df.loc[
            (df['Close'] > df['BB_Upper']) & 
            (df['Close'].shift(1) <= df['BB_Upper'].shift(1)),
            'signal'
        ] = -1
        
        return df


class MACrossStrategy(BaseStrategy):
    """
    Moving Average Trend Following Strategy.
    
    Buy when price is above EMA and EMA is rising.
    Sell when price is below EMA or EMA starts falling.
    """
    
    def __init__(self, short_period: int = 10, long_period: int = 30):
        super().__init__(name=f"MA_Trend_{short_period}_{long_period}")
        self.short_period = short_period
        self.long_period = long_period
    
    def generate_signals(self) -> pd.DataFrame:
        self._ensure_data()
        df = self.data.copy()
        
        # Calculate EMAs
        df['EMA_Short'] = df['Close'].ewm(span=self.short_period, adjust=False).mean()
        df['EMA_Long'] = df['Close'].ewm(span=self.long_period, adjust=False).mean()
        
        # Calculate EMA slope
        df['EMA_Short_Slope'] = df['EMA_Short'].diff()
        df['EMA_Long_Slope'] = df['EMA_Long'].diff()
        
        # Generate signals
        df['signal'] = 0
        
        # Buy: Short EMA above Long EMA and both rising
        buy_condition = (
            (df['EMA_Short'] > df['EMA_Long']) &
            (df['EMA_Short_Slope'] > 0) &
            (df['EMA_Long_Slope'] > 0)
        )
        
        # Sell: Short EMA below Long EMA or either falling
        sell_condition = (
            (df['EMA_Short'] < df['EMA_Long']) |
            (df['EMA_Long_Slope'] < 0)
        )
        
        # Only signal on transitions
        df.loc[buy_condition & ~buy_condition.shift(1).fillna(False), 'signal'] = 1
        df.loc[sell_condition & ~sell_condition.shift(1).fillna(False), 'signal'] = -1
        
        return df


class MomentumStrategy(BaseStrategy):
    """
    Momentum Strategy based on rate of change.
    
    Buy when momentum crosses above threshold (bullish).
    Sell when momentum crosses below threshold (bearish).
    """
    
    def __init__(self, period: int = 20, threshold: float = 0.05):
        super().__init__(name=f"Momentum_{period}")
        self.period = period
        self.threshold = threshold
    
    def generate_signals(self) -> pd.DataFrame:
        self._ensure_data()
        df = self.data.copy()
        
        # Calculate momentum (rate of change)
        df['Momentum'] = df['Close'].pct_change(periods=self.period)
        
        # Generate signals
        df['signal'] = 0
        
        # Buy when momentum crosses above threshold
        df.loc[
            (df['Momentum'] > self.threshold) & 
            (df['Momentum'].shift(1) <= self.threshold),
            'signal'
        ] = 1
        
        # Sell when momentum crosses below negative threshold
        df.loc[
            (df['Momentum'] < -self.threshold) & 
            (df['Momentum'].shift(1) >= -self.threshold),
            'signal'
        ] = -1
        
        return df


# Utility functions for indicators
def calculate_sma(data: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return data.rolling(window=period).mean()


def calculate_ema(data: pd.Series, period: int) -> pd.Series:
    """Calculate Exponential Moving Average."""
    return data.ewm(span=period, adjust=False).mean()


def calculate_rsi(data: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = data.diff()
    gain = delta.where(delta > 0, 0)
    loss = (-delta).where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calculate_bollinger_bands(
    data: pd.Series, 
    period: int = 20, 
    num_std: float = 2.0
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate Bollinger Bands (upper, middle, lower)."""
    middle = data.rolling(window=period).mean()
    std = data.rolling(window=period).std()
    upper = middle + (std * num_std)
    lower = middle - (std * num_std)
    return upper, middle, lower


def calculate_atr(
    high: pd.Series, 
    low: pd.Series, 
    close: pd.Series, 
    period: int = 14
) -> pd.Series:
    """Calculate Average True Range."""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window=period).mean()


def calculate_macd(
    data: pd.Series, 
    fast: int = 12, 
    slow: int = 26, 
    signal: int = 9
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    """Calculate MACD (macd line, signal line, histogram)."""
    ema_fast = data.ewm(span=fast, adjust=False).mean()
    ema_slow = data.ewm(span=slow, adjust=False).mean()
    
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    histogram = macd_line - signal_line
    
    return macd_line, signal_line, histogram
