import pandas as pd
import numpy as np
from typing import Optional, Tuple
from pathlib import Path


class DataHandler:
    """Handles loading and preprocessing of historical trading data."""
    
    # Standard column names for OHLCV data
    REQUIRED_COLUMNS = ['Open', 'High', 'Low', 'Close']
    OPTIONAL_COLUMNS = ['Volume', 'Date']
    
    # Alternative column name mappings (case-insensitive)
    COLUMN_MAPPINGS = {
        'open': 'Open', 'high': 'High', 'low': 'Low', 
        'close': 'Close', 'volume': 'Volume', 'date': 'Date',
        'adj close': 'Adj_Close', 'adj_close': 'Adj_Close'
    }
    
    def __init__(self):
        self.data: Optional[pd.DataFrame] = None
        self.source_file: Optional[str] = None
    
    def load_csv(self, filepath: str, date_column: str = 'Date') -> pd.DataFrame:
        """
        Load OHLCV data from a CSV file.
        
        Args:
            filepath: Path to the CSV file
            date_column: Name of the date column
            
        Returns:
            DataFrame with standardized OHLCV columns
        """
        self.source_file = filepath
        
        # Load the CSV
        df = pd.read_csv(filepath)
        
        # Standardize column names
        df = self._standardize_columns(df)
        
        # Parse dates
        df = self._parse_dates(df, date_column)
        
        # Validate data
        df = self._validate_ohlcv(df)
        
        # Sort by date
        df = df.sort_values('Date').reset_index(drop=True)
        
        self.data = df
        return df
    
    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to expected format."""
        # Create copy to avoid modifying original
        df = df.copy()
        
        # Rename columns to standard format
        new_columns = {}
        for col in df.columns:
            col_lower = col.strip().lower()
            if col_lower in self.COLUMN_MAPPINGS:
                new_columns[col] = self.COLUMN_MAPPINGS[col_lower]
        
        df = df.rename(columns=new_columns)
        
        # Check for required columns
        missing = set(self.REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
        
        return df
    
    def _parse_dates(self, df: pd.DataFrame, date_column: str) -> pd.DataFrame:
        """Parse date column to datetime."""
        df = df.copy()
        
        # Try to find date column
        if date_column in df.columns:
            df['Date'] = pd.to_datetime(df[date_column])
        elif 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date'])
        else:
            # Use first column as date if no Date column found
            first_col = df.columns[0]
            df['Date'] = pd.to_datetime(df[first_col])
        
        return df
    
    def _validate_ohlcv(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validate and clean OHLCV data."""
        df = df.copy()
        
        # Ensure required columns are numeric
        for col in self.REQUIRED_COLUMNS:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Validate OHLC relationships (High >= Low, etc.)
        valid = (
            (df['High'] >= df['Low']) &
            (df['High'] >= df['Open']) &
            (df['High'] >= df['Close']) &
            (df['Low'] <= df['Open']) &
            (df['Low'] <= df['Close'])
        )
        
        # Remove invalid rows
        invalid_count = (~valid).sum()
        if invalid_count > 0:
            print(f"Warning: Removing {invalid_count} rows with invalid OHLC data")
            df = df[valid].copy()
        
        # Handle missing values
        df = df.dropna(subset=self.REQUIRED_COLUMNS)
        
        # Ensure Volume is numeric if present
        if 'Volume' in df.columns:
            df['Volume'] = pd.to_numeric(df['Volume'], errors='coerce').fillna(0)
        
        return df
    
    def add_features(self) -> pd.DataFrame:
        """
        Add technical analysis features to the data.
        
        Returns:
            DataFrame with added feature columns
        """
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv first.")
        
        df = self.data.copy()
        
        # Returns
        df['Returns'] = df['Close'].pct_change()
        df['Log_Returns'] = np.log(df['Close'] / df['Close'].shift(1))
        
        # Price changes
        df['Price_Change'] = df['Close'] - df['Close'].shift(1)
        
        # Volatility (rolling std of returns)
        df['Volatility_5'] = df['Returns'].rolling(window=5).std()
        df['Volatility_20'] = df['Returns'].rolling(window=20).std()
        
        # Average True Range (ATR)
        df = self._add_atr(df)
        
        self.data = df
        return df
    
    def _add_atr(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
        """Calculate Average True Range."""
        high = df['High']
        low = df['Low']
        prev_close = df['Close'].shift(1)
        
        tr1 = high - low
        tr2 = (high - prev_close).abs()
        tr3 = (low - prev_close).abs()
        
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(window=period).mean()
        
        return df
    
    def get_data(self) -> pd.DataFrame:
        """Get the current data."""
        if self.data is None:
            raise ValueError("No data loaded. Call load_csv first.")
        return self.data.copy()
    
    @staticmethod
    def generate_synthetic_data(
        n_days: int = 252,
        initial_price: float = 100.0,
        volatility: float = 0.02,
        drift: float = 0.0001
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data for testing.
        
        Args:
            n_days: Number of trading days to generate
            initial_price: Starting price
            volatility: Daily volatility
            drift: Daily drift (trend)
            
        Returns:
            DataFrame with synthetic OHLCV data
        """
        np.random.seed(42)
        
        # Generate random returns
        returns = np.random.normal(drift, volatility, n_days)
        
        # Generate close prices
        close_prices = initial_price * np.exp(np.cumsum(returns))
        
        # Generate OHLC from close (simplified)
        data = []
        for i, close in enumerate(close_prices):
            # Add some noise to create OHLC
            high_factor = 1 + np.random.uniform(0.001, 0.02)
            low_factor = 1 - np.random.uniform(0.001, 0.02)
            open_factor = 1 + np.random.uniform(-0.005, 0.005)
            
            high = close * high_factor
            low = close * low_factor
            open_price = close * open_factor
            
            # Ensure OHLC integrity
            high = max(high, close, open_price)
            low = min(low, close, open_price)
            
            # Generate volume (random with some patterns)
            base_volume = 1000000
            volume = int(base_volume * np.random.uniform(0.5, 2.0))
            
            data.append({
                'Date': pd.Timestamp('2020-01-01') + pd.Timedelta(days=i),
                'Open': open_price,
                'High': high,
                'Low': low,
                'Close': close,
                'Volume': volume
            })
        
        df = pd.DataFrame(data)
        
        # Add day of week for realism
        df['DayOfWeek'] = df['Date'].dt.dayofweek
        
        # Remove weekends for more realistic data
        df = df[df['DayOfWeek'] < 5].reset_index(drop=True)
        
        return df


def load_data(filepath: str) -> pd.DataFrame:
    """Convenience function to load data from a CSV file."""
    handler = DataHandler()
    return handler.load_csv(filepath)
