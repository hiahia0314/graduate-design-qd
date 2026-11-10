from datetime import datetime
from decimal import Decimal

from src import AccountSnapshot, Bar, MarketSnapshot, Position, Side, TargetPortfolio
from src.core import TargetWeightPortfolio


DATE = datetime(2026, 1, 5)


def bar(symbol: str, close: str) -> Bar:
    price = Decimal(close)
    return Bar(symbol, DATE, price, price, price, price, Decimal("1000000"))


def test_target_weights_create_lot_rounded_sell_first_orders() -> None:
    market = MarketSnapshot(
        DATE,
        {
            "000001.SZ": bar("000001.SZ", "10"),
            "600000.SH": bar("600000.SH", "20"),
            "601318.SH": bar("601318.SH", "40"),
        },
    )
    account = AccountSnapshot(
        timestamp=DATE,
        cash=Decimal("200000"),
        total_equity=Decimal("1000000"),
        positions={"601318.SH": Position("601318.SH", 1000, Decimal("38"))},
    )
    target = TargetPortfolio(
        target_id="TGT-1",
        strategy_id="demo",
        timestamp=DATE,
        weights={"000001.SZ": Decimal("0.5"), "600000.SH": Decimal("0.5")},
    )

    orders = TargetWeightPortfolio().create_orders(target, account, market)

    assert [order.side for order in orders] == [Side.SELL, Side.BUY, Side.BUY]
    assert [(order.symbol, order.quantity) for order in orders] == [
        ("601318.SH", 1000),
        ("000001.SZ", 50000),
        ("600000.SH", 25000),
    ]
    assert all(order.quantity % 100 == 0 for order in orders)
