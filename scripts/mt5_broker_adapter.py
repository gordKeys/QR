from datetime import datetime, timezone
from functools import lru_cache


class MT5UnavailableError(RuntimeError):
    pass


class MT5BrokerAdapter:

    def __init__(self, terminal_path=None):
        try:
            import MetaTrader5 as mt5  # type: ignore
        except Exception as exc:
            raise MT5UnavailableError(
                "MetaTrader5 package is not installed in this environment."
            ) from exc

        self.mt5 = mt5
        self.terminal_path = terminal_path

    def initialize(self):
        if self.terminal_path:
            initialized = self.mt5.initialize(path=self.terminal_path)
        else:
            initialized = self.mt5.initialize()
        if not initialized:
            raise RuntimeError(f"MT5 initialize failed: {self.mt5.last_error()}")

    def shutdown(self):
        self.mt5.shutdown()

    def account_equity(self):
        info = self.mt5.account_info()
        return info.equity if info else None

    def positions_total(self, symbol=None):
        positions = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
        return 0 if positions is None else len(positions)

    def positions_get(self, symbol=None):
        positions = self.mt5.positions_get(symbol=symbol) if symbol else self.mt5.positions_get()
        return [] if positions is None else list(positions)

    def symbol_info(self, symbol):
        return self.mt5.symbol_info(symbol)

    def symbol_tick(self, symbol):
        return self.mt5.symbol_info_tick(symbol)

    def symbols_get(self):
        symbols = self.mt5.symbols_get()
        return [] if symbols is None else list(symbols)

    @lru_cache(maxsize=256)
    def resolve_symbol(self, symbol):
        target = symbol.upper()
        symbols = self.symbols_get()
        if not symbols:
            return None

        by_upper = {getattr(item, "name", "").upper(): getattr(item, "name", "") for item in symbols}
        if target in by_upper:
            return by_upper[target]

        prefix_matches = []
        for item in symbols:
            name = getattr(item, "name", "")
            upper_name = name.upper()
            if upper_name == target:
                return name
            if upper_name.startswith(target):
                suffix = upper_name[len(target):]
                if len(suffix) <= 8:
                    prefix_matches.append(name)

        if not prefix_matches:
            return None

        def score(name):
            info = self.symbol_info(name)
            suffix = name.upper()[len(target):]
            trade_mode = getattr(info, "trade_mode", 0) if info is not None else 0
            visible = 1 if getattr(info, "visible", False) else 0
            return (
                len(suffix),
                0 if visible else 1,
                0 if trade_mode else 1,
                0 if suffix and suffix[0] in ".-_#" else 1,
                name,
            )

        return sorted(prefix_matches, key=score)[0]

    def rates_copy(self, symbol, timeframe, count=500):
        return self.mt5.copy_rates_from_pos(symbol, timeframe, 0, count)

    def rates_range(self, symbol, timeframe, date_from, date_to):
        return self.mt5.copy_rates_range(symbol, timeframe, date_from, date_to)

    def order_calc_margin(self, direction, symbol, volume, price):
        order_type = self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL
        return self.mt5.order_calc_margin(order_type, symbol, volume, price)

    def normalize_volume(self, symbol, volume):
        info = self.symbol_info(symbol)
        if info is None:
            return volume

        min_volume = getattr(info, "volume_min", 0.01) or 0.01
        max_volume = getattr(info, "volume_max", volume) or volume
        step = getattr(info, "volume_step", 0.01) or 0.01

        clipped = max(min_volume, min(volume, max_volume))
        steps = round(clipped / step)
        normalized = steps * step
        return max(min_volume, round(normalized, 8))

    def normalize_price(self, symbol, price):
        info = self.symbol_info(symbol)
        if info is None:
            return price

        digits = getattr(info, "digits", None)
        if digits is None:
            return price

        return round(price, int(digits))

    def min_stop_distance(self, symbol):
        info = self.symbol_info(symbol)
        if info is None:
            return 0.0

        stops_level = float(getattr(info, "trade_stops_level", 0.0) or 0.0)
        point = float(getattr(info, "point", 0.0) or 0.0)
        return stops_level * point

    def conform_stop_levels(self, symbol, direction, entry_price, stop_loss, take_profit):
        min_distance = self.min_stop_distance(symbol)
        if min_distance > 0:
            if direction == 1:
                stop_loss = min(stop_loss, entry_price - min_distance)
                take_profit = max(take_profit, entry_price + min_distance)
            else:
                stop_loss = max(stop_loss, entry_price + min_distance)
                take_profit = min(take_profit, entry_price - min_distance)

        stop_loss = self.normalize_price(symbol, stop_loss)
        take_profit = self.normalize_price(symbol, take_profit)
        entry_price = self.normalize_price(symbol, entry_price)

        if direction == 1 and not (stop_loss < entry_price < take_profit):
            return None, None
        if direction == -1 and not (take_profit < entry_price < stop_loss):
            return None, None

        return stop_loss, take_profit

    def filling_modes(self, symbol):
        info = self.symbol_info(symbol)
        if info is None:
            return [
                self.mt5.ORDER_FILLING_IOC,
                self.mt5.ORDER_FILLING_FOK,
                self.mt5.ORDER_FILLING_RETURN,
            ]

        supported = getattr(info, "filling_mode", 0) or 0
        modes = []

        if supported & 2:
            modes.append(self.mt5.ORDER_FILLING_IOC)
        if supported & 1:
            modes.append(self.mt5.ORDER_FILLING_FOK)

        for mode in (
            self.mt5.ORDER_FILLING_IOC,
            self.mt5.ORDER_FILLING_FOK,
            self.mt5.ORDER_FILLING_RETURN,
        ):
            if mode not in modes:
                modes.append(mode)

        return modes

    def history_deals_since(self, since_time, symbol=None, magic=None):
        deals = self.mt5.history_deals_get(since_time, datetime.now(timezone.utc))
        if deals is None:
            return []

        filtered = []
        for deal in deals:
            if symbol is not None and getattr(deal, "symbol", None) != symbol:
                continue
            if magic is not None and getattr(deal, "magic", None) != magic:
                continue
            filtered.append(deal)
        return filtered

    def place_order(self, *, symbol, direction, volume, stop_loss, take_profit, price=None, comment="QuantFX"):
        order_type = self.mt5.ORDER_TYPE_BUY if direction == 1 else self.mt5.ORDER_TYPE_SELL
        if price is None:
            tick = self.mt5.symbol_info_tick(symbol)
            price = tick.ask if direction == 1 else tick.bid

        price = self.normalize_price(symbol, price)
        stop_loss = self.normalize_price(symbol, stop_loss)
        take_profit = self.normalize_price(symbol, take_profit)

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": price,
            "sl": stop_loss,
            "tp": take_profit,
            "deviation": 20,
            "magic": 26072026,
            "comment": comment,
            "type_time": self.mt5.ORDER_TIME_GTC,
        }

        last_result = None
        invalid_fill = getattr(self.mt5, "TRADE_RETCODE_INVALID_FILL", 10030)
        for fill_mode in self.filling_modes(symbol):
            request["type_filling"] = fill_mode
            last_result = self.mt5.order_send(request)
            if last_result is not None and getattr(last_result, "retcode", None) == self.mt5.TRADE_RETCODE_DONE:
                return last_result
            if last_result is not None and getattr(last_result, "retcode", None) != invalid_fill:
                break

        return last_result

    def modify_position_stops(self, *, position_ticket, symbol, stop_loss=None, take_profit=None, comment="QuantFX"):
        request = {
            "action": self.mt5.TRADE_ACTION_SLTP,
            "position": int(position_ticket),
            "symbol": symbol,
            "magic": 26072026,
            "comment": comment,
        }
        if stop_loss is not None:
            request["sl"] = self.normalize_price(symbol, stop_loss)
        if take_profit is not None:
            request["tp"] = self.normalize_price(symbol, take_profit)
        return self.mt5.order_send(request)

    def close_position(self, position, *, comment="QuantFX close"):
        tick = self.symbol_tick(position.symbol)
        if tick is None:
            return None

        direction = getattr(position, "type", None)
        buy_type = getattr(self.mt5, "POSITION_TYPE_BUY", 0)
        sell_type = getattr(self.mt5, "POSITION_TYPE_SELL", 1)

        if direction == buy_type:
            close_type = self.mt5.ORDER_TYPE_SELL
            price = tick.bid
        elif direction == sell_type:
            close_type = self.mt5.ORDER_TYPE_BUY
            price = tick.ask
        else:
            return None

        request = {
            "action": self.mt5.TRADE_ACTION_DEAL,
            "position": int(getattr(position, "ticket", 0) or 0),
            "symbol": position.symbol,
            "volume": float(getattr(position, "volume", 0.0) or 0.0),
            "type": close_type,
            "price": self.normalize_price(position.symbol, price),
            "deviation": 20,
            "magic": int(getattr(position, "magic", 26072026) or 26072026),
            "comment": comment,
            "type_time": self.mt5.ORDER_TIME_GTC,
        }

        last_result = None
        invalid_fill = getattr(self.mt5, "TRADE_RETCODE_INVALID_FILL", 10030)
        for fill_mode in self.filling_modes(position.symbol):
            request["type_filling"] = fill_mode
            last_result = self.mt5.order_send(request)
            if last_result is not None and getattr(last_result, "retcode", None) == self.mt5.TRADE_RETCODE_DONE:
                return last_result
            if last_result is not None and getattr(last_result, "retcode", None) != invalid_fill:
                break

        return last_result
