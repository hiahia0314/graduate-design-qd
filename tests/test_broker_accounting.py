from datetime import datetime, timedelta
from decimal import Decimal

from src import Bar, MarketSnapshot, Order, Position, Side
from src.core import EquityAccounting, NextOpenBroker


DATE = datetime(2026, 1, 6)


def make_bar(symbol: str, price: str) -> Bar:
    value = Decimal(price)
    return Bar(symbol, DATE, value, value, value, value, Decimal("1000"))


def make_order(order_id: str, symbol: str, side: Side, quantity: int) -> Order:
    return Order(order_id, "TGT-1", "demo", symbol, DATE - timedelta(days=1), side, quantity)


def test_broker_executes_sells_before_buys() -> None:
    market = MarketSnapshot(
        DATE,
        {"SELL.SH": make_bar("SELL.SH", "20"), "BUY.SZ": make_bar("BUY.SZ", "10")},
    )
    broker = NextOpenBroker()
    broker.submit(
        [
            make_order("BUY-1", "BUY.SZ", Side.BUY, 100),
            make_order("SELL-1", "SELL.SH", Side.SELL, 100),
        ]
    )

    fills = broker.match(market, Decimal("0"))

    assert [fill.side for fill in fills] == [Side.SELL, Side.BUY]
    assert fills[1].quantity == 100


def test_broker_scales_buys_proportionally_without_negative_cash() -> None:
    market = MarketSnapshot(
        DATE,
        {"A.SZ": make_bar("A.SZ", "10"), "B.SH": make_bar("B.SH", "10")},
    )
    broker = NextOpenBroker()
    broker.submit(
        [
            make_order("A", "A.SZ", Side.BUY, 1000),
            make_order("B", "B.SH", Side.BUY, 1000),
        ]
    )

    fills = broker.match(market, Decimal("10000"))

    assert [fill.quantity for fill in fills] == [500, 500]
    assert sum(fill.price * fill.quantity for fill in fills) == Decimal("10000")


def test_accounting_updates_cash_positions_and_daily_equity() -> None:
    market = MarketSnapshot(DATE, {"A.SZ": make_bar("A.SZ", "10")})
    broker = NextOpenBroker()
    broker.submit([make_order("A", "A.SZ", Side.BUY, 100)])
    fill = broker.match(market, Decimal("10000"))[0]
    accounting = EquityAccounting(Decimal("10000"))

    accounting.on_fill(fill)
    snapshot = accounting.mark(market)

    assert accounting.cash == Decimal("9000.00")
    assert snapshot.positions["A.SZ"].quantity == 100
    assert snapshot.total_equity == Decimal("10000.00")


def test_accounting_uses_last_close_when_held_symbol_has_no_current_bar() -> None:
    first_market = MarketSnapshot(DATE, {"A.SZ": make_bar("A.SZ", "10")})
    broker = NextOpenBroker()
    broker.submit([make_order("A", "A.SZ", Side.BUY, 100)])
    fill = broker.match(first_market, Decimal("10000"))[0]
    accounting = EquityAccounting(Decimal("10000"))

    accounting.on_fill(fill)
    accounting.mark(first_market)
    missing_bar_market = MarketSnapshot(DATE + timedelta(days=1), {})

    snapshot = accounting.mark(missing_bar_market)

    assert snapshot.total_equity == Decimal("10000.00")
    assert snapshot.positions["A.SZ"].quantity == 100


def test_broker_rejects_same_day_execution_without_consuming_order():
    import pytest
    from dataclasses import replace
    broker = NextOpenBroker()
    order = replace(make_order("A", "A.SZ", Side.BUY, 100), timestamp=DATE)
    broker.submit([order])
    with pytest.raises(ValueError, match="later trading date"):
        broker.match(MarketSnapshot(DATE, {"A.SZ": make_bar("A.SZ", "10")}), Decimal("10000"))
    next_day = DATE + timedelta(days=1)
    fills = broker.match(MarketSnapshot(next_day, {"A.SZ": replace(make_bar("A.SZ", "10"), timestamp=next_day)}), Decimal("10000"))
    assert len(fills) == 1


def test_accounting_rejects_overdraw_and_oversell_atomically():
    import pytest
    from src import Fill, FillStatus
    account = EquityAccounting(Decimal("100"))
    fill = Fill("f", "o", "t", "s", FillStatus.FILLED, "A", DATE, Side.BUY, 100, Decimal("10"))
    with pytest.raises(ValueError, match="negative"):
        account.on_fill(fill)
    from dataclasses import replace
    with pytest.raises(ValueError, match="more shares"):
        account.on_fill(replace(fill, side=Side.SELL))
    assert account.cash == Decimal("100")
    assert account.positions == {}


def test_blocked_orders_cancel_and_do_not_consume_buy_budget():
    from dataclasses import replace
    from src.framework.models import OrderStatus
    broker = NextOpenBroker()
    broker.submit([
        make_order("blocked", "A", Side.BUY, 100),
        make_order("ok", "B", Side.BUY, 100),
        make_order("missing", "C", Side.BUY, 100),
    ])
    market = MarketSnapshot(DATE, {
        "A": replace(make_bar("A", "10"), buy_block_reason="limit_up_at_open"),
        "B": make_bar("B", "10"),
    })
    fills = broker.match(market, Decimal("1000"))
    assert [(f.order_id, f.quantity) for f in fills] == [("ok", 100)]
    outcomes = {o.order_id: o for o in broker.outcomes}
    assert outcomes["blocked"].reason == "limit_up_at_open"
    assert outcomes["blocked"].status is OrderStatus.CANCELLED
    assert outcomes["missing"].reason == "missing_bar"
    assert broker.pending_orders == ()
    assert broker.match(replace(market, timestamp=DATE + timedelta(days=1)), Decimal("1000")) == []


def test_partial_fill_cancels_remainder_with_quantity_conservation():
    from src.framework.models import OrderStatus
    broker = NextOpenBroker()
    broker.submit([make_order("a", "A", Side.BUY, 1000)])
    broker.match(MarketSnapshot(DATE, {"A": make_bar("A", "10")}), Decimal("2500"))
    outcome = broker.outcomes[0]
    assert outcome.status is OrderStatus.PARTIAL_CANCELLED
    assert outcome.filled_quantity == 200
    assert outcome.cancelled_quantity == 800
    assert outcome.requested_quantity == outcome.filled_quantity + outcome.cancelled_quantity


def test_sell_block_does_not_release_cash_for_buys():
    from dataclasses import replace
    broker = NextOpenBroker()
    broker.submit([make_order("sell", "A", Side.SELL, 100), make_order("buy", "B", Side.BUY, 100)])
    market = MarketSnapshot(DATE, {
        "A": replace(make_bar("A", "10"), sell_block_reason="suspended_at_open"),
        "B": make_bar("B", "10"),
    })
    assert broker.match(market, Decimal("0")) == []
    assert broker.outcomes[0].reason == "suspended_at_open"
    assert broker.outcomes[1].cancelled_quantity == 100
