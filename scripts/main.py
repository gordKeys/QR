from engine.data_loader import DataLoader
from engine.features import FeatureEngine
from engine.regime import RegimeDetector
from engine.strategy_selector import StrategySelector
from engine.multi_backtester import MultiStrategyBacktester
from engine.performance_report import PerformanceReport

from strategies.mean_reversion import MeanReversion
from strategies.momentum import Momentum
from strategies.trend_follow import TrendFollowing
from strategies.volatility_breakout import VolatilityBreakout


# ===================
# LOAD DATA
# ===================
loader = DataLoader("data/EURUSD_M5.csv")
data = loader.load()

print("\n=== RAW DATA ===")
print("Shape:", data.shape)
print("Columns:", data.columns)


# ===================
# FEATURES (LOCKED PIPELINE STEP 1)
# ===================
data = FeatureEngine().add_features(data)

print("\n=== FEATURES ADDED ===")
print(data.columns)


# ===================
# REGIME DETECTION (LOCKED PIPELINE STEP 2)
# ===================
data = RegimeDetector().detect(data)

print("\n=== REGIME ADDED ===")
print(data.columns)


# ===================
# STRATEGIES
# ===================
strategies = {
    "mean_reversion": MeanReversion(),
    "momentum": Momentum(),
    "trend": TrendFollowing(),
    "volatility_breakout": VolatilityBreakout()
}

selector = StrategySelector()


# ===================
# BACKTEST
# ===================
bt = MultiStrategyBacktester(
    data=data,
    strategies=strategies,
    selector=selector
)

results = bt.run()


# ===================
# REPORTING
# ===================
print("\n=== DATA CHECK ===")
print(data.head())
print("\nShape:", data.shape)
print("Time Range:", data.index.min(), "→", data.index.max())


PerformanceReport(results).full_report()

print("\n=== FINAL SUMMARY ===")
print("Final Balance:", results["final_balance"])
print("Trades:", results["total_trades"])
print("Win Rate:", results["win_rate"])