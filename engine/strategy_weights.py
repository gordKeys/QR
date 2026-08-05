import pandas as pd


class StrategyRegimeWeights:

    def __init__(self):

        # base learned weights (can later be ML trained)
        self.weights = {
            "momentum": {
                -1: 0.6,
                 1: 0.8
            },
            "trend": {
                -1: 0.3,
                 1: 0.9
            },
            "mean_reversion": {
                -1: 0.7,
                 1: 0.4
            }
        }

    def get_weight(self, strategy: str, regime: int):

        return self.weights.get(strategy, {}).get(regime, 0.5)