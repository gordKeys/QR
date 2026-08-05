class StrategySelector:

    def select(self, df, i):

        regime = df["regime"].iloc[i]

        if regime == 1:
            return "trend"
        elif regime == -1:
            return "mean_reversion"
        elif regime == 2:
            return "volatility_breakout"
        else:
            return "momentum"
