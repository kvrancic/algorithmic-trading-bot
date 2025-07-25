"""
Technical Features for QuantumSentiment Trading Bot

Comprehensive technical analysis features including:
- Price action indicators
- Momentum indicators
- Volume indicators
- Volatility indicators
- Trend indicators
- Support/Resistance levels
"""

from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
import talib
from scipy import stats
from scipy.signal import find_peaks
import structlog

logger = structlog.get_logger(__name__)

# Note: pandas_ta has numpy compatibility issues, using TA-Lib primarily
try:
    import pandas_ta as ta
except ImportError:
    logger.warning("pandas_ta not available, using TA-Lib only")
    ta = None


class TechnicalFeatures:
    """Technical analysis feature generator"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize technical features generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        
        # Default parameters
        self.default_params = {
            'sma_periods': [5, 10, 20, 50, 100, 200],
            'ema_periods': [5, 10, 20, 50, 100, 200],
            'rsi_periods': [14, 21, 30],
            'bb_period': 20,
            'bb_std': 2,
            'macd_fast': 12,
            'macd_slow': 26,
            'macd_signal': 9,
            'atr_period': 14,
            'adx_period': 14,
            'stoch_k': 14,
            'stoch_d': 3,
            'williams_period': 14,
            'cci_period': 20,
            'momentum_periods': [5, 10, 20],
            'roc_periods': [5, 10, 20],
            'volume_sma_periods': [10, 20, 50]
        }
        
        # Merge with provided config
        self.params = {**self.default_params, **self.config}
        
        logger.info("Technical features initialized")
    
    def generate_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Generate all technical features from OHLCV data
        
        Args:
            df: DataFrame with OHLCV data (indexed by timestamp)
            
        Returns:
            Dictionary of technical features
        """
        if df.empty or len(df) < 50:
            logger.warning("Insufficient data for technical features", records=len(df))
            return {}
        
        try:
            features = {}
            
            # Ensure we have required columns
            required_cols = ['open', 'high', 'low', 'close', 'volume']
            if not all(col in df.columns for col in required_cols):
                logger.error("Missing required OHLCV columns", 
                           available=list(df.columns))
                return {}
            
            # Extract series for faster computation
            # Ensure correct data types for TA-Lib (float64)
            open_prices = np.asarray(df['open'].values, dtype=np.float64)
            high_prices = np.asarray(df['high'].values, dtype=np.float64)
            low_prices = np.asarray(df['low'].values, dtype=np.float64)
            close_prices = np.asarray(df['close'].values, dtype=np.float64)
            volume = np.asarray(df['volume'].values, dtype=np.float64)
            
            # Basic price features
            features.update(self._price_features(close_prices, open_prices, high_prices, low_prices))
            
            # Moving averages
            features.update(self._moving_average_features(close_prices))
            
            # Momentum indicators
            features.update(self._momentum_features(close_prices, high_prices, low_prices))
            
            # Volatility indicators
            features.update(self._volatility_features(close_prices, high_prices, low_prices))
            
            # Volume indicators
            features.update(self._volume_features(close_prices, high_prices, low_prices, volume))
            
            # Trend indicators
            features.update(self._trend_features(close_prices, high_prices, low_prices))
            
            # Support/Resistance features
            features.update(self._support_resistance_features(close_prices, high_prices, low_prices))
            
            # Candlestick patterns
            features.update(self._candlestick_patterns(open_prices, high_prices, low_prices, close_prices))
            
            # Statistical features
            features.update(self._statistical_features(close_prices))
            
            # Market structure features
            features.update(self._market_structure_features(df))
            
            logger.debug("Technical features generated", count=len(features))
            return features
            
        except Exception as e:
            logger.error("Failed to generate technical features", error=str(e))
            return {}
    
    def _price_features(self, close: np.ndarray, open_: np.ndarray, 
                       high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
        """Basic price action features"""
        features = {}
        
        try:
            current_price = close[-1]
            
            # Returns
            features['return_1d'] = (close[-1] / close[-2] - 1) if len(close) > 1 else 0
            features['return_5d'] = (close[-1] / close[-6] - 1) if len(close) > 5 else 0
            features['return_20d'] = (close[-1] / close[-21] - 1) if len(close) > 20 else 0
            
            # Log returns
            if len(close) > 1:
                features['log_return_1d'] = np.log(close[-1] / close[-2])
                
            # Price position within recent range
            recent_high = np.max(high[-20:]) if len(high) >= 20 else np.max(high)
            recent_low = np.min(low[-20:]) if len(low) >= 20 else np.min(low)
            
            if recent_high != recent_low:
                features['price_position_20d'] = (current_price - recent_low) / (recent_high - recent_low)
            
            # Gap features
            if len(close) > 1:
                features['gap_up'] = max(0, (open_[-1] - close[-2]) / close[-2])
                features['gap_down'] = max(0, (close[-2] - open_[-1]) / close[-2])
            
            # Intraday range
            if high[-1] != low[-1]:
                features['intraday_range'] = (high[-1] - low[-1]) / close[-1]
                features['close_position_in_range'] = (close[-1] - low[-1]) / (high[-1] - low[-1])
            
        except Exception as e:
            logger.warning("Error in price features", error=str(e))
        
        return features
    
    def _moving_average_features(self, close: np.ndarray) -> Dict[str, float]:
        """Moving average features"""
        features = {}
        
        try:
            current_price = close[-1]
            
            # Simple Moving Averages
            for period in self.params['sma_periods']:
                if len(close) >= period:
                    sma = talib.SMA(close, timeperiod=period)
                    if not np.isnan(sma[-1]):
                        features[f'sma_{period}'] = sma[-1]
                        features[f'price_vs_sma_{period}'] = (current_price / sma[-1] - 1)
                        
                        # SMA slope
                        if len(sma) >= 5 and not np.isnan(sma[-5]):
                            features[f'sma_{period}_slope'] = (sma[-1] - sma[-5]) / sma[-5]
            
            # Exponential Moving Averages
            for period in self.params['ema_periods']:
                if len(close) >= period:
                    ema = talib.EMA(close, timeperiod=period)
                    if not np.isnan(ema[-1]):
                        features[f'ema_{period}'] = ema[-1]
                        features[f'price_vs_ema_{period}'] = (current_price / ema[-1] - 1)
            
            # Moving average convergence/divergence
            if len(close) >= 50:
                sma_10 = talib.SMA(close, timeperiod=10)
                sma_20 = talib.SMA(close, timeperiod=20)
                sma_50 = talib.SMA(close, timeperiod=50)
                
                if not (np.isnan(sma_10[-1]) or np.isnan(sma_20[-1]) or np.isnan(sma_50[-1])):
                    features['sma_10_vs_20'] = (sma_10[-1] / sma_20[-1] - 1)
                    features['sma_20_vs_50'] = (sma_20[-1] / sma_50[-1] - 1)
                    
                    # Golden cross / death cross
                    features['golden_cross'] = 1 if sma_10[-1] > sma_20[-1] else 0
                    features['death_cross'] = 1 if sma_10[-1] < sma_20[-1] else 0
            
        except Exception as e:
            logger.warning("Error in moving average features", error=str(e))
        
        return features
    
    def _momentum_features(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
        """Momentum indicator features"""
        features = {}
        
        try:
            # RSI
            for period in self.params['rsi_periods']:
                if len(close) >= period + 1:
                    rsi = talib.RSI(close, timeperiod=period)
                    if not np.isnan(rsi[-1]):
                        features[f'rsi_{period}'] = rsi[-1]
                        
                        # RSI conditions
                        features[f'rsi_{period}_overbought'] = 1 if rsi[-1] > 70 else 0
                        features[f'rsi_{period}_oversold'] = 1 if rsi[-1] < 30 else 0
            
            # MACD
            if len(close) >= self.params['macd_slow'] + self.params['macd_signal']:
                macd, macd_signal, macd_hist = talib.MACD(
                    close, 
                    fastperiod=self.params['macd_fast'],
                    slowperiod=self.params['macd_slow'],
                    signalperiod=self.params['macd_signal']
                )
                
                if not np.isnan(macd[-1]):
                    features['macd'] = macd[-1]
                    features['macd_signal'] = macd_signal[-1]
                    features['macd_histogram'] = macd_hist[-1]
                    features['macd_bullish'] = 1 if macd[-1] > macd_signal[-1] else 0
            
            # Stochastic Oscillator
            if len(close) >= self.params['stoch_k']:
                slowk, slowd = talib.STOCH(
                    high, low, close,
                    fastk_period=self.params['stoch_k'],
                    slowk_period=self.params['stoch_d'],
                    slowd_period=self.params['stoch_d']
                )
                
                if not np.isnan(slowk[-1]):
                    features['stoch_k'] = slowk[-1]
                    features['stoch_d'] = slowd[-1]
                    features['stoch_overbought'] = 1 if slowk[-1] > 80 else 0
                    features['stoch_oversold'] = 1 if slowk[-1] < 20 else 0
            
            # Williams %R
            if len(close) >= self.params['williams_period']:
                williams_r = talib.WILLR(high, low, close, timeperiod=self.params['williams_period'])
                if not np.isnan(williams_r[-1]):
                    features['williams_r'] = williams_r[-1]
            
            # Commodity Channel Index
            if len(close) >= self.params['cci_period']:
                cci = talib.CCI(high, low, close, timeperiod=self.params['cci_period'])
                if not np.isnan(cci[-1]):
                    features['cci'] = cci[-1]
                    features['cci_overbought'] = 1 if cci[-1] > 100 else 0
                    features['cci_oversold'] = 1 if cci[-1] < -100 else 0
            
            # Momentum
            for period in self.params['momentum_periods']:
                if len(close) >= period + 1:
                    momentum = talib.MOM(close, timeperiod=period)
                    if not np.isnan(momentum[-1]):
                        features[f'momentum_{period}'] = momentum[-1]
            
            # Rate of Change
            for period in self.params['roc_periods']:
                if len(close) >= period + 1:
                    roc = talib.ROC(close, timeperiod=period)
                    if not np.isnan(roc[-1]):
                        features[f'roc_{period}'] = roc[-1]
            
        except Exception as e:
            logger.warning("Error in momentum features", error=str(e))
        
        return features
    
    def _volatility_features(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
        """Volatility indicator features"""
        features = {}
        
        try:
            # Average True Range
            if len(close) >= self.params['atr_period'] + 1:
                atr = talib.ATR(high, low, close, timeperiod=self.params['atr_period'])
                if not np.isnan(atr[-1]):
                    features['atr'] = atr[-1]
                    features['atr_pct'] = atr[-1] / close[-1]
            
            # Bollinger Bands
            if len(close) >= self.params['bb_period']:
                bb_upper, bb_middle, bb_lower = talib.BBANDS(
                    close, 
                    timeperiod=self.params['bb_period'],
                    nbdevup=self.params['bb_std'],
                    nbdevdn=self.params['bb_std']
                )
                
                if not np.isnan(bb_upper[-1]):
                    features['bb_upper'] = bb_upper[-1]
                    features['bb_middle'] = bb_middle[-1]
                    features['bb_lower'] = bb_lower[-1]
                    
                    # Bollinger Band position
                    bb_width = bb_upper[-1] - bb_lower[-1]
                    if bb_width > 0:
                        features['bb_position'] = (close[-1] - bb_lower[-1]) / bb_width
                        features['bb_squeeze'] = 1 if bb_width < np.mean(bb_upper[-20:] - bb_lower[-20:]) else 0
            
            # Historical volatility
            if len(close) >= 21:
                returns = np.diff(np.log(close))
                features['volatility_20d'] = np.std(returns[-20:]) * np.sqrt(252)
                features['volatility_10d'] = np.std(returns[-10:]) * np.sqrt(252)
                
            # Keltner Channels
            if len(close) >= 20:
                kc_middle = talib.EMA(close, timeperiod=20)
                atr_20 = talib.ATR(high, low, close, timeperiod=20)
                
                if not (np.isnan(kc_middle[-1]) or np.isnan(atr_20[-1])):
                    kc_upper = kc_middle + (2 * atr_20)
                    kc_lower = kc_middle - (2 * atr_20)
                    
                    features['kc_position'] = (close[-1] - kc_lower[-1]) / (kc_upper[-1] - kc_lower[-1])
            
        except Exception as e:
            logger.warning("Error in volatility features", error=str(e))
        
        return features
    
    def _volume_features(self, close: np.ndarray, high: np.ndarray, low: np.ndarray, volume: np.ndarray) -> Dict[str, float]:
        """Volume-based features"""
        features = {}
        
        try:
            # Ensure data types are correct for TA-Lib (float64)
            close = np.asarray(close, dtype=np.float64)
            high = np.asarray(high, dtype=np.float64)
            low = np.asarray(low, dtype=np.float64)
            volume = np.asarray(volume, dtype=np.float64)
            current_volume = volume[-1]
            
            # Volume moving averages
            for period in self.params['volume_sma_periods']:
                if len(volume) >= period:
                    vol_sma = talib.SMA(volume, timeperiod=period)
                    if not np.isnan(vol_sma[-1]):
                        features[f'volume_sma_{period}'] = vol_sma[-1]
                        features[f'volume_vs_sma_{period}'] = current_volume / vol_sma[-1]
            
            # On Balance Volume
            if len(close) >= 20:
                obv = talib.OBV(close, volume)
                if not np.isnan(obv[-1]):
                    features['obv'] = obv[-1]
                    
                    # OBV trend
                    if len(obv) >= 10:
                        obv_slope = (obv[-1] - obv[-10]) / obv[-10] if obv[-10] != 0 else 0
                        features['obv_slope'] = obv_slope
            
            # Volume Price Trend
            if len(close) >= 10:
                vpt = talib.TRIX(volume, timeperiod=10)  # Using TRIX as proxy
                if not np.isnan(vpt[-1]):
                    features['vpt'] = vpt[-1]
            
            # Accumulation/Distribution Line
            if len(close) >= 10:
                ad = talib.AD(high, low, close, volume)
                if not np.isnan(ad[-1]):
                    features['ad_line'] = ad[-1]
            
            # Chaikin Money Flow
            if len(close) >= 20 and ta is not None:
                try:
                    cmf = ta.cmf(high, low, close, volume, length=20)
                    if cmf is not None and not cmf.empty:
                        features['cmf'] = cmf.iloc[-1]
                except Exception as e:
                    logger.debug("CMF calculation failed", error=str(e))
            
            # Volume statistics
            if len(volume) >= 20:
                vol_mean = np.mean(volume[-20:])
                vol_std = np.std(volume[-20:])
                
                features['volume_zscore'] = (current_volume - vol_mean) / vol_std if vol_std > 0 else 0
                features['volume_high'] = 1 if current_volume > vol_mean + 2*vol_std else 0
            
        except Exception as e:
            logger.warning("Error in volume features", error=str(e))
        
        return features
    
    def _trend_features(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
        """Trend identification features"""
        features = {}
        
        try:
            # Average Directional Index (ADX)
            if len(close) >= self.params['adx_period'] + 1:
                adx = talib.ADX(high, low, close, timeperiod=self.params['adx_period'])
                plus_di = talib.PLUS_DI(high, low, close, timeperiod=self.params['adx_period'])
                minus_di = talib.MINUS_DI(high, low, close, timeperiod=self.params['adx_period'])
                
                if not np.isnan(adx[-1]):
                    features['adx'] = adx[-1]
                    features['plus_di'] = plus_di[-1]
                    features['minus_di'] = minus_di[-1]
                    features['trend_strength'] = 1 if adx[-1] > 25 else 0
                    features['trend_direction'] = 1 if plus_di[-1] > minus_di[-1] else -1
            
            # Parabolic SAR
            if len(close) >= 10:
                sar = talib.SAR(high, low, acceleration=0.02, maximum=0.2)
                if not np.isnan(sar[-1]):
                    features['sar'] = sar[-1]
                    features['price_vs_sar'] = 1 if close[-1] > sar[-1] else -1
            
            # Linear regression slope
            if len(close) >= 20:
                x = np.arange(len(close[-20:]))
                slope, intercept, r_value, p_value, std_err = stats.linregress(x, close[-20:])
                features['lr_slope_20'] = slope / close[-1]  # Normalized slope
                features['lr_r_squared_20'] = r_value ** 2
            
            # Trend consistency
            if len(close) >= 10:
                short_trend = np.mean(close[-5:]) - np.mean(close[-10:-5])
                long_trend = np.mean(close[-10:]) - np.mean(close[-20:-10]) if len(close) >= 20 else 0
                
                features['trend_consistency'] = 1 if short_trend * long_trend > 0 else 0
            
        except Exception as e:
            logger.warning("Error in trend features", error=str(e))
        
        return features
    
    def _support_resistance_features(self, close: np.ndarray, high: np.ndarray, low: np.ndarray) -> Dict[str, float]:
        """Support and resistance level features"""
        features = {}
        
        try:
            current_price = close[-1]
            
            # Find recent highs and lows
            if len(high) >= 20:
                peaks, _ = find_peaks(high[-50:] if len(high) >= 50 else high, distance=5)
                troughs, _ = find_peaks(-low[-50:] if len(low) >= 50 else -low, distance=5)
                
                if len(peaks) > 0:
                    recent_resistance = high[-50:][peaks[-1]] if len(high) >= 50 else high[peaks[-1]]
                    features['resistance_distance'] = (recent_resistance - current_price) / current_price
                
                if len(troughs) > 0:
                    recent_support = low[-50:][troughs[-1]] if len(low) >= 50 else low[troughs[-1]]
                    features['support_distance'] = (current_price - recent_support) / current_price
            
            # Fibonacci retracements
            if len(close) >= 50:
                recent_high = np.max(high[-50:])
                recent_low = np.min(low[-50:])
                
                if recent_high != recent_low:
                    fib_levels = [0.236, 0.382, 0.5, 0.618, 0.786]
                    for fib in fib_levels:
                        fib_level = recent_high - (recent_high - recent_low) * fib
                        distance = abs(current_price - fib_level) / current_price
                        features[f'fib_{int(fib*1000)}_distance'] = distance
            
            # Round number proximity
            round_numbers = [50, 100, 200, 500, 1000]
            for rn in round_numbers:
                if current_price > rn * 0.1:  # Only consider relevant round numbers
                    nearest_round = round(current_price / rn) * rn
                    features[f'round_{rn}_distance'] = abs(current_price - nearest_round) / current_price
            
        except Exception as e:
            logger.warning("Error in support/resistance features", error=str(e))
        
        return features
    
    def _candlestick_patterns(self, open_: np.ndarray, high: np.ndarray, 
                            low: np.ndarray, close: np.ndarray) -> Dict[str, float]:
        """Candlestick pattern recognition features"""
        features = {}
        
        try:
            if len(close) < 5:
                return features
            
            # Major candlestick patterns
            patterns = {
                'doji': talib.CDLDOJI,
                'hammer': talib.CDLHAMMER,
                'hanging_man': talib.CDLHANGINGMAN,
                'inverted_hammer': talib.CDLINVERTEDHAMMER,
                'shooting_star': talib.CDLSHOOTINGSTAR,
                'engulfing_bullish': talib.CDLENGULFING,
                'harami': talib.CDLHARAMI,
                'dark_cloud': talib.CDLDARKCLOUDCOVER,
                'piercing': talib.CDLPIERCING,
                'morning_star': talib.CDLMORNINGSTAR,
                'evening_star': talib.CDLEVENINGSTAR,
                'three_white_soldiers': talib.CDL3WHITESOLDIERS,
                'three_black_crows': talib.CDL3BLACKCROWS
            }
            
            for pattern_name, pattern_func in patterns.items():
                try:
                    pattern_result = pattern_func(open_, high, low, close)
                    if not np.isnan(pattern_result[-1]):
                        features[f'pattern_{pattern_name}'] = 1 if pattern_result[-1] != 0 else 0
                except:
                    features[f'pattern_{pattern_name}'] = 0
            
            # Custom pattern features
            features['green_candle'] = 1 if close[-1] > open_[-1] else 0
            features['red_candle'] = 1 if close[-1] < open_[-1] else 0
            
            # Body size
            body_size = abs(close[-1] - open_[-1]) / close[-1]
            features['candle_body_size'] = body_size
            features['large_body'] = 1 if body_size > 0.02 else 0  # 2% body
            
            # Wick analysis
            upper_wick = (high[-1] - max(open_[-1], close[-1])) / close[-1]
            lower_wick = (min(open_[-1], close[-1]) - low[-1]) / close[-1]
            
            features['upper_wick_size'] = upper_wick
            features['lower_wick_size'] = lower_wick
            features['long_upper_wick'] = 1 if upper_wick > 0.01 else 0
            features['long_lower_wick'] = 1 if lower_wick > 0.01 else 0
            
        except Exception as e:
            logger.warning("Error in candlestick pattern features", error=str(e))
        
        return features
    
    def _statistical_features(self, close: np.ndarray) -> Dict[str, float]:
        """Statistical features based on price series"""
        features = {}
        
        try:
            if len(close) < 20:
                return features
            
            # Z-score features
            for window in [10, 20, 50]:
                if len(close) >= window:
                    window_mean = np.mean(close[-window:])
                    window_std = np.std(close[-window:])
                    
                    if window_std > 0:
                        features[f'zscore_{window}'] = (close[-1] - window_mean) / window_std
            
            # Skewness and kurtosis
            if len(close) >= 30:
                returns = np.diff(np.log(close[-30:]))
                features['returns_skewness'] = stats.skew(returns)
                features['returns_kurtosis'] = stats.kurtosis(returns)
            
            # Percentile ranks
            for window in [20, 50]:
                if len(close) >= window:
                    percentile_rank = stats.percentileofscore(close[-window:], close[-1])
                    features[f'percentile_rank_{window}'] = percentile_rank / 100
            
            # Autocorrelation
            if len(close) >= 20:
                returns = np.diff(np.log(close[-20:]))
                if len(returns) > 1:
                    autocorr = np.corrcoef(returns[:-1], returns[1:])[0, 1]
                    features['returns_autocorr'] = autocorr if not np.isnan(autocorr) else 0
            
        except Exception as e:
            logger.warning("Error in statistical features", error=str(e))
        
        return features
    
    def _market_structure_features(self, df: pd.DataFrame) -> Dict[str, float]:
        """Market microstructure features"""
        features = {}
        
        try:
            if len(df) < 10:
                return features
            
            close = df['close'].values
            high = df['high'].values
            low = df['low'].values
            volume = df['volume'].values
            
            # Price efficiency measures
            if len(close) >= 10:
                # Variance ratio test statistic
                returns = np.diff(np.log(close[-20:] if len(close) >= 20 else close))
                if len(returns) >= 4:
                    var_1 = np.var(returns)
                    # Calculate 2-period returns properly by summing consecutive returns
                    returns_2 = returns[:-1] + returns[1:]  # Sum consecutive pairs
                    var_2 = np.var(returns_2) / 2  # 2-period returns variance
                    
                    if var_1 > 0:
                        features['variance_ratio'] = var_2 / var_1
            
            # Bid-ask spread proxy (high-low spread)
            if len(df) >= 5:
                hl_spreads = (high - low) / close
                features['avg_hl_spread'] = np.mean(hl_spreads[-5:])
                features['hl_spread_volatility'] = np.std(hl_spreads[-10:]) if len(df) >= 10 else 0
            
            # Volume-price relationship
            if len(df) >= 10:
                price_changes = np.diff(close[-10:])
                volume_changes = np.diff(volume[-10:])
                
                # Correlation between price changes and volume
                if len(price_changes) > 1 and np.std(price_changes) > 0 and np.std(volume_changes) > 0:
                    pv_corr = np.corrcoef(price_changes, volume_changes)[0, 1]
                    features['price_volume_corr'] = pv_corr if not np.isnan(pv_corr) else 0
            
            # Market impact proxy
            if len(df) >= 5:
                price_impacts = np.abs(np.diff(close[-5:])) / volume[-5:-1]
                price_impacts = price_impacts[price_impacts != np.inf]  # Remove infinite values
                
                if len(price_impacts) > 0:
                    features['avg_price_impact'] = np.mean(price_impacts)
            
        except Exception as e:
            logger.warning("Error in market structure features", error=str(e))
        
        return features
    
    def get_feature_names(self) -> List[str]:
        """Get list of all possible feature names"""
        # This would return all possible feature names
        # Implementation would generate a comprehensive list
        feature_names = []
        
        # Add all possible feature names based on configuration
        # This is a simplified version - full implementation would be more comprehensive
        
        return feature_names
    
    def __str__(self) -> str:
        return f"TechnicalFeatures(params={len(self.params)})"
    
    def __repr__(self) -> str:
        return self.__str__()