from dataclasses import dataclass
from datetime import datetime


@dataclass
class Trade:

    direction: int              # 1 = Buy, -1 = Sell

    entry_price: float
    stop_loss: float
    take_profit: float

    position_size: float

    strategy: str
    regime: int

    entry_time: datetime

    exit_price: float = None
    exit_time: datetime = None

    pnl: float = 0.0
    r_multiple: float = 0.0

    status: str = "OPEN"