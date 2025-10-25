#!/usr/bin/env python3
"""
Trend Analysis for Stock Prediction System

This module handles trend analysis including moving averages calculation,
trend detection, reversal point identification, and technical analysis
according to the system specifications.
"""

import numpy as np
import pandas as pd
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
from scipy.signal import find_peaks, argrelextrema
from scipy.stats import linregress
import warnings

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)


class TrendAnalyzer:
    """
    Trend Analysis class for stock prediction system
    
    Handles calculation of moving averages, trend detection, and reversal analysis
    according to system specifications:
    - Short trend: 5-day moving average slope
    - Long trend: 25-day moving average slope
    - Reversal prices: Peak/trough prices at trend switches
    """
    
    def __init__(self, config):
        """
        Initialize TrendAnalyzer
        
        Args:
            config: Configuration object containing trend analysis settings
        """
        self.config = config
        self.short_window = config.model.short_term_window  # 5 days
        self.long_window = config.model.long_term_window    # 25 days
        self.trend_threshold = config.prediction.trend_threshold
        self.reversal_sensitivity = config.prediction.reversal_sensitivity
        self.volatility_window = config.prediction.volatility_window
        
        logger.info("TrendAnalyzer initialized")
    
    def _calculate_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate basic financial features
    
    Args:
        df (pd.DataFrame): Stock data
        
    Returns:
        pd.DataFrame: Data with basic features
    """
    # Daily returns (1-day percentage change)
    df['return_1d'] = df['close'].pct_change()
    
    # 5-day volatility (standard deviation of returns)
    df['volatility_5d'] = df['return_1d'].rolling(window=5, min_periods=5).std()
    
    # 20-day volatility
    df['volatility_20d'] = df['return_1d'].rolling(window=20, min_periods=20).std()
    
    return df

def _calculate_rsi(self, df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Relative Strength Index (RSI)
    
    Args:
        df (pd.DataFrame): Stock data
        period (int): RSI period (default 14)
        
    Returns:
        pd.DataFrame: Data with RSI
    """
    # Calculate price changes
    delta = df['close'].diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period, min_periods=period).mean()
    avg_losses = losses.rolling(window=period, min_periods=period).mean()
    
    # Calculate RS and RSI
    rs = avg_gains / avg_losses
    df['rsi'] = 100 - (100 / (1 + rs))
    
    # Fill NaN with 50 (neutral RSI)
    df['rsi'] = df['rsi'].fillna(50)
    
    return df

def _calculate_macd(self, df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Args:
        df (pd.DataFrame): Stock data
        fast (int): Fast EMA period (default 12)
        slow (int): Slow EMA period (default 26)
        signal (int): Signal line period (default 9)
        
    Returns:
        pd.DataFrame: Data with MACD and signal line
    """
    # Calculate EMAs
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    # Calculate MACD line
    df['macd'] = ema_fast - ema_slow
    
    # Calculate signal line
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    
    # Fill NaN with 0
    df['macd'] = df['macd'].fillna(0)
    df['macd_signal'] = df['macd_signal'].fillna(0)
    
    return df

def _calculate_all_technical_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators in one pass
    This is a helper method that combines all feature calculations
    
    Args:
        df (pd.DataFrame): Stock data
        
    Returns:
        pd.DataFrame: Data with all technical indicators
    """
    # Basic features (returns and volatility)
    df = self._calculate_basic_features(df)
    
    # RSI
    df = self._calculate_rsi(df)
    
    # MACD
    df = self._calculate_macd(df)
    
    return df
    
    def calculate_indicators(self, data: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate all technical indicators for a single stock's data
    
    Args:
        data (pd.DataFrame): Stock price data
        
    Returns:
        pd.DataFrame: Data with all calculated indicators
    """
    try:
        df = data.copy()
        
        # Calculate all technical indicators FIRST
        df = self._calculate_all_technical_indicators(df)
        
        # Calculate moving averages
        df[f'ma_{self.short_window}'] = df['close'].rolling(
            window=self.short_window, min_periods=self.short_window
        ).mean()
        
        df[f'ma_{self.long_window}'] = df['close'].rolling(
            window=self.long_window, min_periods=self.long_window
        ).mean()
        
        # Calculate additional moving averages
        df['ma_10'] = df['close'].rolling(window=10, min_periods=10).mean()
        df['ma_50'] = df['close'].rolling(window=50, min_periods=50).mean()
        
        # Calculate exponential moving averages
        df[f'ema_{self.short_window}'] = df['close'].ewm(span=self.short_window).mean()
        df[f'ema_{self.long_window}'] = df['close'].ewm(span=self.long_window).mean()
        
        # Calculate trend slopes
        df = self._calculate_trend_slopes(df)
        
        # Calculate trend directions
        df = self._calculate_trend_directions(df)
        
        # Calculate trend strength
        df = self._calculate_trend_strength(df)
        
        # Calculate trend consistency
        df = self._calculate_trend_consistency(df)
        
        # Calculate volatility features
        df = self._calculate_volatility_features(df)
        
        # Calculate momentum indicators
        df = self._calculate_momentum_indicators(df)
        
        return df
        
    except Exception as e:
        logger.warning(f"Error calculating indicators: {e}")
        return data

    def calculate_moving_averages(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Calculate moving averages for all stocks
        
        Args:
            data (Dict[str, pd.DataFrame]): Stock price data
            
        Returns:
            Dict[str, pd.DataFrame]: Data with moving averages
        """
        processed_data = {}
        
        for stock_code, stock_data in data.items():
            try:
                df = stock_data.copy()
                
                # Calculate short-term moving average (5-day)
                df[f'ma_{self.short_window}'] = df['close'].rolling(
                    window=self.short_window, min_periods=self.short_window
                ).mean()
                
                # Calculate long-term moving average (25-day)
                df[f'ma_{self.long_window}'] = df['close'].rolling(
                    window=self.long_window, min_periods=self.long_window
                ).mean()
                
                # Calculate additional moving averages for context
                df['ma_10'] = df['close'].rolling(window=10, min_periods=10).mean()
                df['ma_50'] = df['close'].rolling(window=50, min_periods=50).mean()
                
                # Calculate exponential moving averages
                df[f'ema_{self.short_window}'] = df['close'].ewm(span=self.short_window).mean()
                df[f'ema_{self.long_window}'] = df['close'].ewm(span=self.long_window).mean()
                
                processed_data[stock_code] = df
                
            except Exception as e:
                logger.warning(f"Error calculating moving averages for {stock_code}: {e}")
        
        logger.info(f"Calculated moving averages for {len(processed_data)} stocks")
        return processed_data
    
    def calculate_trend_features(self, data: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
    """
    Calculate trend features including slopes, directions, and strength
    
    Args:
        data (Dict[str, pd.DataFrame]): Data with moving averages
        
    Returns:
        Dict[str, pd.DataFrame]: Data with trend features
    """
    processed_data = {}
    
    for stock_code, stock_data in data.items():
        try:
            df = stock_data.copy()
            
            # Calculate all technical indicators FIRST
            df = self._calculate_all_technical_indicators(df)
            
            # Calculate trend slopes
            df = self._calculate_trend_slopes(df)
            
            # Calculate trend directions
            df = self._calculate_trend_directions(df)
            
            # Calculate trend strength
            df = self._calculate_trend_strength(df)
            
            # Calculate trend consistency
            df = self._calculate_trend_consistency(df)
            
            # Calculate volatility features
            df = self._calculate_volatility_features(df)
            
            # Calculate momentum indicators
            df = self._calculate_momentum_indicators(df)
            
            processed_data[stock_code] = df
            
        except Exception as e:
            logger.warning(f"Error calculating trend features for {stock_code}: {e}")
    
    logger.info(f"Calculated trend features for {len(processed_data)} stocks")
    return processed_data
    
    def _calculate_trend_slopes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate slopes of moving averages (trend indicators)
        
        Args:
            df (pd.DataFrame): Data with moving averages
            
        Returns:
            pd.DataFrame: Data with trend slopes
        """
        # Short-term trend slope (5-day MA)
        ma_short_col = f'ma_{self.short_window}'
        if ma_short_col in df.columns:
            # Calculate slope using linear regression over a small window
            df['short_trend_slope'] = df[ma_short_col].rolling(window=3).apply(
                lambda x: self._calculate_slope(x) if len(x) == 3 else np.nan
            )
            
            # Simple difference-based slope
            df['short_trend_slope_simple'] = df[ma_short_col].diff()
        
        # Long-term trend slope (25-day MA)
        ma_long_col = f'ma_{self.long_window}'
        if ma_long_col in df.columns:
            # Calculate slope using linear regression over a small window
            df['long_trend_slope'] = df[ma_long_col].rolling(window=5).apply(
                lambda x: self._calculate_slope(x) if len(x) == 5 else np.nan
            )
            
            # Simple difference-based slope
            df['long_trend_slope_simple'] = df[ma_long_col].diff()
        
        return df
    
    def _calculate_slope(self, values: pd.Series) -> float:
        """
        Calculate slope using linear regression
        
        Args:
            values (pd.Series): Values to calculate slope for
            
        Returns:
            float: Slope value
        """
        if len(values) < 2:
            return np.nan
        
        x = np.arange(len(values))
        y = values.values
        
        # Remove NaN values
        mask = ~np.isnan(y)
        if mask.sum() < 2:
            return np.nan
        
        try:
            slope, _, _, _, _ = linregress(x[mask], y[mask])
            return slope
        except:
            return np.nan
    
    def _calculate_trend_directions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend directions based on slopes
        
        Args:
            df (pd.DataFrame): Data with trend slopes
            
        Returns:
            pd.DataFrame: Data with trend directions
        """
        # Short-term trend direction
        if 'short_trend_slope' in df.columns:
            df['short_trend_direction'] = np.where(
                df['short_trend_slope'] > self.trend_threshold, 1,  # Upward
                np.where(df['short_trend_slope'] < -self.trend_threshold, -1, 0)  # Downward or Sideways
            )
            
            # Convert to +/- format as specified
            df['short_trend'] = np.where(
                df['short_trend_direction'] == 1, '+',
                np.where(df['short_trend_direction'] == -1, '-', '0')
            )
        
        # Long-term trend direction
        if 'long_trend_slope' in df.columns:
            df['long_trend_direction'] = np.where(
                df['long_trend_slope'] > self.trend_threshold, 1,  # Upward
                np.where(df['long_trend_slope'] < -self.trend_threshold, -1, 0)  # Downward or Sideways
            )
            
            # Convert to +/- format as specified
            df['long_trend'] = np.where(
                df['long_trend_direction'] == 1, '+',
                np.where(df['long_trend_direction'] == -1, '-', '0')
            )
        
        return df
    
    def _calculate_trend_strength(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend strength indicators
        
        Args:
            df (pd.DataFrame): Data with trend information
            
        Returns:
            pd.DataFrame: Data with trend strength
        """
        # Short-term trend strength
        if 'short_trend_slope' in df.columns:
            df['short_trend_strength'] = abs(df['short_trend_slope'])
        
        # Long-term trend strength
        if 'long_trend_slope' in df.columns:
            df['long_trend_strength'] = abs(df['long_trend_slope'])
        
        # Trend agreement (when short and long trends align)
        if 'short_trend_direction' in df.columns and 'long_trend_direction' in df.columns:
            df['trend_agreement'] = (
                df['short_trend_direction'] * df['long_trend_direction']
            ).clip(lower=0)  # 1 when aligned, 0 when not
        
        return df
    
    def _calculate_trend_consistency(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate trend consistency over time
        
        Args:
            df (pd.DataFrame): Data with trend directions
            
        Returns:
            pd.DataFrame: Data with trend consistency
        """
        # Short-term trend consistency (how often trend direction stays the same)
        if 'short_trend_direction' in df.columns:
            df['short_trend_consistency'] = df['short_trend_direction'].rolling(
                window=10
            ).apply(lambda x: (x == x.iloc[-1]).sum() / len(x) if len(x) > 0 else np.nan)
        
        # Long-term trend consistency
        if 'long_trend_direction' in df.columns:
            df['long_trend_consistency'] = df['long_trend_direction'].rolling(
                window=20
            ).apply(lambda x: (x == x.iloc[-1]).sum() / len(x) if len(x) > 0 else np.nan)
        
        return df
    
    def _calculate_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate volatility-based features
        
        Args:
            df (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with volatility features
        """
        # Price volatility
        df['price_volatility'] = df['close'].rolling(
            window=self.volatility_window
        ).std()
        
        # Moving average volatility
        ma_short_col = f'ma_{self.short_window}'
        if ma_short_col in df.columns:
            df['ma_short_volatility'] = df[ma_short_col].rolling(
                window=self.volatility_window
            ).std()
        
        ma_long_col = f'ma_{self.long_window}'
        if ma_long_col in df.columns:
            df['ma_long_volatility'] = df[ma_long_col].rolling(
                window=self.volatility_window
            ).std()
        
        # Volatility ratio
        if 'price_volatility' in df.columns:
            df['volatility_ratio'] = df['price_volatility'] / df['price_volatility'].rolling(
                window=50
            ).mean()
        
        return df
    
    def _calculate_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum indicators
        
        Args:
            df (pd.DataFrame): Stock data
            
        Returns:
            pd.DataFrame: Data with momentum indicators
        """
        # Rate of change
        df['roc_5'] = df['close'].pct_change(periods=5)
        df['roc_25'] = df['close'].pct_change(periods=25)
        
        # Moving average convergence/divergence
        ma_short_col = f'ma_{self.short_window}'
        ma_long_col = f'ma_{self.long_window}'
        
        if ma_short_col in df.columns and ma_long_col in df.columns:
            df['ma_convergence'] = df[ma_short_col] - df[ma_long_col]
            df['ma_convergence_pct'] = (df[ma_short_col] / df[ma_long_col] - 1) * 100
        
        return df
    
    def calculate_short_trend(self, data: pd.DataFrame) -> str:
        """
        Calculate current short-term trend for a stock
        
        Args:
            data (pd.DataFrame): Stock price data
            
        Returns:
            str: '+' for upward, '-' for downward, '0' for sideways
        """
        try:
            if len(data) < self.short_window:
                return '0'
            
            # Calculate 5-day moving average
            ma_short = data['close'].rolling(window=self.short_window).mean()
            
            # Get recent values for slope calculation
            recent_ma = ma_short.dropna().tail(3)
            
            if len(recent_ma) < 2:
                return '0'
            
            # Calculate slope
            slope = self._calculate_slope(recent_ma)
            
            if pd.isna(slope):
                return '0'
            
            # Determine trend direction
            if slope > self.trend_threshold:
                return '+'
            elif slope < -self.trend_threshold:
                return '-'
            else:
                return '0'
                
        except Exception as e:
            logger.warning(f"Error calculating short trend: {e}")
            return '0'
    
    def calculate_long_trend(self, data: pd.DataFrame) -> str:
        """
        Calculate current long-term trend for a stock
        
        Args:
            data (pd.DataFrame): Stock price data
            
        Returns:
            str: '+' for upward, '-' for downward, '0' for sideways
        """
        try:
            if len(data) < self.long_window:
                return '0'
            
            # Calculate 25-day moving average
            ma_long = data['close'].rolling(window=self.long_window).mean()
            
            # Get recent values for slope calculation
            recent_ma = ma_long.dropna().tail(5)
            
            if len(recent_ma) < 2:
                return '0'
            
            # Calculate slope
            slope = self._calculate_slope(recent_ma)
            
            if pd.isna(slope):
                return '0'
            
            # Determine trend direction
            if slope > self.trend_threshold:
                return '+'
            elif slope < -self.trend_threshold:
                return '-'
            else:
                return '0'
                
        except Exception as e:
            logger.warning(f"Error calculating long trend: {e}")
            return '0'
    
    def find_trend_reversals(self, data: pd.DataFrame, trend_type: str = 'short') -> List[Dict[str, Any]]:
        """
        Find trend reversal points in the data
        
        Args:
            data (pd.DataFrame): Stock data with moving averages
            trend_type (str): 'short' or 'long' trend analysis
            
        Returns:
            List[Dict[str, Any]]: List of reversal points with details
        """
        try:
            window = self.short_window if trend_type == 'short' else self.long_window
            ma_col = f'ma_{window}'
            
            if ma_col not in data.columns:
                # Calculate moving average if not present
                data = data.copy()
                data[ma_col] = data['close'].rolling(window=window).mean()
            
            # Get moving average values
            ma_values = data[ma_col].dropna()
            
            if len(ma_values) < window * 2:
                return []
            
            # Find peaks and troughs
            peaks, _ = find_peaks(ma_values.values, distance=window//2)
            troughs, _ = find_peaks(-ma_values.values, distance=window//2)
            
            reversals = []
            
            # Process peaks (downward reversals)
            for peak_idx in peaks:
                actual_idx = ma_values.index[peak_idx]
                reversal = {
                    'date': data.loc[actual_idx, 'date'],
                    'index': actual_idx,
                    'type': 'peak',
                    'price': ma_values.iloc[peak_idx],
                    'actual_price': data.loc[actual_idx, 'close'],
                    'trend_before': '+',
                    'trend_after': '-'
                }
                reversals.append(reversal)
            
            # Process troughs (upward reversals)
            for trough_idx in troughs:
                actual_idx = ma_values.index[trough_idx]
                reversal = {
                    'date': data.loc[actual_idx, 'date'],
                    'index': actual_idx,
                    'type': 'trough',
                    'price': ma_values.iloc[trough_idx],
                    'actual_price': data.loc[actual_idx, 'close'],
                    'trend_before': '-',
                    'trend_after': '+'
                }
                reversals.append(reversal)
            
            # Sort by date
            reversals.sort(key=lambda x: x['date'])
            
            return reversals
            
        except Exception as e:
            logger.warning(f"Error finding trend reversals: {e}")
            return []
    
    def predict_next_reversal_price(self, data: pd.DataFrame, trend_type: str = 'short') -> Dict[str, Any]:
        """
        Predict the next trend reversal price
        
        Args:
            data (pd.DataFrame): Stock data
            trend_type (str): 'short' or 'long' trend analysis
            
        Returns:
            Dict[str, Any]: Predicted reversal information
        """
        try:
            # Find historical reversals
            reversals = self.find_trend_reversals(data, trend_type)
            
            if len(reversals) < 2:
                return {'price': data['close'].iloc[-1], 'confidence': 0.0, 'type': 'unknown'}
            
            # Get current trend
            current_trend = (self.calculate_short_trend(data) if trend_type == 'short' 
                           else self.calculate_long_trend(data))
            
            # Get recent reversals for pattern analysis
            recent_reversals = reversals[-5:]  # Last 5 reversals
            
            # Calculate average reversal magnitude
            reversal_magnitudes = []
            for i in range(1, len(recent_reversals)):
                prev_reversal = recent_reversals[i-1]
                curr_reversal = recent_reversals[i]
                magnitude = abs(curr_reversal['price'] - prev_reversal['price'])
                reversal_magnitudes.append(magnitude)
            
            if not reversal_magnitudes:
                return {'price': data['close'].iloc[-1], 'confidence': 0.0, 'type': 'unknown'}
            
            avg_magnitude = np.mean(reversal_magnitudes)
            current_price = data['close'].iloc[-1]
            
            # Predict next reversal based on current trend
            if current_trend == '+':
                # Upward trend, predict peak (downward reversal)
                predicted_price = current_price + avg_magnitude * 0.5
                reversal_type = 'peak'
            elif current_trend == '-':
                # Downward trend, predict trough (upward reversal)
                predicted_price = current_price - avg_magnitude * 0.5
                reversal_type = 'trough'
            else:
                # Sideways trend, use recent reversal as reference
                predicted_price = recent_reversals[-1]['price']
                reversal_type = 'sideways'
            
            # Calculate confidence based on trend consistency and reversal pattern regularity
            confidence = self._calculate_reversal_confidence(data, reversals, trend_type)
            
            return {
                'price': predicted_price,
                'confidence': confidence,
                'type': reversal_type,
                'current_trend': current_trend,
                'avg_magnitude': avg_magnitude
            }
            
        except Exception as e:
            logger.warning(f"Error predicting reversal price: {e}")
            return {'price': data['close'].iloc[-1], 'confidence': 0.0, 'type': 'error'}
    
    def _calculate_reversal_confidence(self, data: pd.DataFrame, reversals: List[Dict], trend_type: str) -> float:
        """
        Calculate confidence in reversal prediction
        
        Args:
            data (pd.DataFrame): Stock data
            reversals (List[Dict]): Historical reversals
            trend_type (str): Trend type
            
        Returns:
            float: Confidence score (0-1)
        """
        try:
            if len(reversals) < 3:
                return 0.3
            
            confidence_factors = []
            
            # Factor 1: Regularity of reversal intervals
            intervals = []
            for i in range(1, len(reversals)):
                interval = (reversals[i]['date'] - reversals[i-1]['date']).days
                intervals.append(interval)
            
            if intervals:
                interval_std = np.std(intervals)
                interval_mean = np.mean(intervals)
                regularity_score = max(0, 1 - (interval_std / max(interval_mean, 1)))
                confidence_factors.append(regularity_score)
            
            # Factor 2: Trend consistency
            window = self.short_window if trend_type == 'short' else self.long_window
            if len(data) >= window:
                recent_data = data.tail(window)
                trend_changes = 0
                prev_trend = None
                
                for _, row in recent_data.iterrows():
                    curr_trend = (self.calculate_short_trend(data.loc[:row.name]) if trend_type == 'short'
                                else self.calculate_long_trend(data.loc[:row.name]))
                    if prev_trend and curr_trend != prev_trend:
                        trend_changes += 1
                    prev_trend = curr_trend
                
                consistency_score = max(0, 1 - (trend_changes / window))
                confidence_factors.append(consistency_score)
            
            # Factor 3: Volatility factor (lower volatility = higher confidence)
            if 'price_volatility' in data.columns:
                recent_volatility = data['price_volatility'].tail(20).mean()
                historical_volatility = data['price_volatility'].mean()
                if historical_volatility > 0:
                    volatility_factor = max(0, 1 - (recent_volatility / historical_volatility))
                    confidence_factors.append(volatility_factor)
            
            # Combine confidence factors
            if confidence_factors:
                return np.mean(confidence_factors)
            else:
                return 0.5
                
        except Exception as e:
            logger.warning(f"Error calculating reversal confidence: {e}")
            return 0.3
    
    def get_trend_summary(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        Get comprehensive trend summary for a stock
        
        Args:
            data (pd.DataFrame): Stock data
            
        Returns:
            Dict[str, Any]: Trend summary
        """
        try:
            # Calculate trends
            short_trend = self.calculate_short_trend(data)
            long_trend = self.calculate_long_trend(data)
            
            # Find reversals
            short_reversals = self.find_trend_reversals(data, 'short')
            long_reversals = self.find_trend_reversals(data, 'long')
            
            # Predict next reversals
            next_short_reversal = self.predict_next_reversal_price(data, 'short')
            next_long_reversal = self.predict_next_reversal_price(data, 'long')
            
            # Calculate additional metrics
            current_price = data['close'].iloc[-1]
            
            summary = {
                'current_price': current_price,
                'short_trend': short_trend,
                'long_trend': long_trend,
                'trend_agreement': short_trend == long_trend,
                'short_reversal_prediction': next_short_reversal,
                'long_reversal_prediction': next_long_reversal,
                'recent_reversals': {
                    'short': short_reversals[-3:] if len(short_reversals) >= 3 else short_reversals,
                    'long': long_reversals[-3:] if len(long_reversals) >= 3 else long_reversals
                },
                'analysis_date': datetime.now().isoformat()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error creating trend summary: {e}")
            return {}

# Example usage and testing
if __name__ == "__main__":
    # For testing purposes
    from config import Config
    from data_manager import DataManager
    
    # Initialize components
    config = Config("test_config.json")
    data_manager = DataManager(config)
    trend_analyzer = TrendAnalyzer(config)
    
    # Load sample data
    stock_list = data_manager.load_stock_list()
    sample_stocks = stock_list['code'].head(3).tolist()
    price_data = data_manager.load_price_data(sample_stocks)
    
    if price_data:
        # Test trend analysis
        sample_stock = list(price_data.keys())[0]
        sample_data = price_data[sample_stock]
        
        print(f"Analyzing trends for stock: {sample_stock}")
        
        # Calculate moving averages
        processed_data = trend_analyzer.calculate_moving_averages({sample_stock: sample_data})
        
        # Calculate trend features
        trend_data = trend_analyzer.calculate_trend_features(processed_data)
        
        # Get trend summary
        summary = trend_analyzer.get_trend_summary(trend_data[sample_stock])
        
        print("Trend Summary:")
        print(f"Current Price: {summary.get('current_price', 'N/A')}")
        print(f"Short Trend: {summary.get('short_trend', 'N/A')}")
        print(f"Long Trend: {summary.get('long_trend', 'N/A')}")
        print(f"Trend Agreement: {summary.get('trend_agreement', 'N/A')}")
        
        if 'short_reversal_prediction' in summary:
            pred = summary['short_reversal_prediction']
            print(f"Short Reversal Prediction: {pred.get('price', 'N/A'):.2f} (confidence: {pred.get('confidence', 0):.2f})")
        
        if 'long_reversal_prediction' in summary:
            pred = summary['long_reversal_prediction']
            print(f"Long Reversal Prediction: {pred.get('price', 'N/A'):.2f} (confidence: {pred.get('confidence', 0):.2f})")
    else:
        print("No sample data available for testing")