from datetime import datetime, timedelta


class VirtualTrader:
    def __init__(self, initial_balance=100000.0):
        self.balance = float(initial_balance)
        self.holdings = 0.0
        self.last_buy_price = None
        self.trade_history = []
        self.net_worth_history = []
        self.last_action = None
        self.cooldown_until = None 

    def buy(self, price, atr=None, confidence=0.0):
        if price is None or price <= 0:
            print("Invalid price for buy.")
            return
        if self.balance <= 0:
            print("Not enough balance to buy.")
            return
        if self.cooldown_until and datetime.utcnow() < self.cooldown_until:
            print("Cooldown active. Cannot buy yet.")
            return

        if confidence < 0.04:
            print(f"Confidence ({confidence:.4f}) too low for trade.")
            return

        base_risk_fraction = 0.05
        max_risk_fraction = 0.40

        risk_fraction = base_risk_fraction + (max_risk_fraction - base_risk_fraction) * confidence
        risk_fraction = min(risk_fraction, max_risk_fraction)

        max_spend = self.balance * risk_fraction

        if atr is not None and atr > 0:
            stop_loss_pct = 0.02
            dollar_risk_per_unit = price * stop_loss_pct + atr
            quantity = max_spend / dollar_risk_per_unit
        else:
            quantity = max_spend / price

        cost = quantity * price

        if cost > max_spend:
            quantity = max_spend / price
            cost = quantity * price

        if cost > self.balance:
            quantity = self.balance / price
            cost = quantity * price

        self.balance -= cost
        self.holdings += quantity
        self.last_buy_price = price

        self.trade_history.append({
            "Type": "BUY",
            "Price": round(price, 2),
            "Quantity": round(quantity, 6),
            "Total": round(cost, 2),
            "PnL": None,
            "Timestamp": datetime.utcnow().isoformat()
        })
        self.last_action = "BUY"

        self.cooldown_until = datetime.utcnow() + timedelta(hours=2)

        print(
            f"[BUY] Bought {quantity:.6f} BTC for ${cost:.2f}. Risk fraction: {risk_fraction * 100:.1f}%. Balance left: ${self.balance:.2f}")

    def sell(self, price, portion=1.0, reason="Auto"):
        if price is None or price <= 0:
            print("Invalid price for sell.")
            return
        if self.holdings <= 0:
            print("No BTC holdings to sell.")
            return

        portion = max(0.0, min(portion, 1.0))
        quantity = self.holdings * portion
        proceeds = quantity * price
        pnl = 0.0

        if self.last_buy_price is not None:
            pnl = (price - self.last_buy_price) * quantity

        self.trade_history.append({
            "Type": "SELL",
            "Price": round(price, 2),
            "Quantity": round(quantity, 6),
            "Total": round(proceeds, 2),
            "PnL": round(pnl, 2),
            "Timestamp": datetime.utcnow().isoformat(),
            "Reason": reason
        })

        self.last_action = "SELL"
        self.balance += proceeds
        self.holdings -= quantity

        if self.holdings <= 0:
            self.holdings = 0.0
            self.last_buy_price = None

        if pnl < 0:
            self.cooldown_until = datetime.utcnow() + timedelta(minutes=10)

        print(f"[SELL] Sold {quantity:.6f} BTC for ${proceeds:.2f}. Reason: {reason}. PnL: ${pnl:.2f}")

    def average_buy_price(self):
        open_buys = []
        net_btc = 0.0

        for trade in self.trade_history:
            qty = float(trade["Quantity"])
            price = float(trade["Price"])

            if trade["Type"] == "BUY":
                open_buys.append([qty, price])
                net_btc += qty
            elif trade["Type"] == "SELL":
                to_sell = qty
                net_btc -= qty
                while to_sell > 0 and open_buys:
                    buy_qty, buy_price = open_buys[0]
                    if buy_qty <= to_sell:
                        to_sell -= buy_qty
                        open_buys.pop(0)
                    else:
                        open_buys[0][0] -= to_sell
                        to_sell = 0

        total_btc = sum(q for q, _ in open_buys)
        total_cost = sum(q * p for q, p in open_buys)
        return (total_cost / total_btc) if total_btc > 0 else None

    def total_trades(self):
        return len(self.trade_history)

    def evaluate(self, price, take_profit=None, stop_loss=None):
        if price is None or price <= 0:
            print("Invalid price for evaluation.")
            return round(self.balance, 2), self.balance, self.holdings

        if self.holdings > 0 and self.last_buy_price:
            change = (price - self.last_buy_price) / self.last_buy_price

            if take_profit is not None and change >= take_profit:
                print("Take-profit hit!")
                self.sell(price, reason="Take Profit")

            elif stop_loss is not None and change <= stop_loss:
                print("Stop-loss hit!")
                self.sell(price, reason="Stop Loss")

            elif change >= 0.02:
                print("Partial profit-taking triggered.")
                self.sell(price, portion=0.25, reason="Partial Take Profit")

            elif change <= -0.03:
                print("Emergency stop-loss!")
                self.sell(price, reason="Emergency Stop Loss")

        net = self.balance + self.holdings * price

        self.net_worth_history.append({
            "timestamp": datetime.utcnow().isoformat(),
            "net_worth": round(net, 2)
        })

        return round(net, 2), self.balance, self.holdings

    def reset(self, initial_balance=100000.0):
        self.__init__(initial_balance)