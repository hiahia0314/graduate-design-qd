from datetime import datetime
from decimal import Decimal


def test_public_api_uses_target_portfolio_instead_of_signal() -> None:
    import src

    assert "TargetPortfolio" in src.__all__
    assert "MarketSnapshot" in src.__all__
    assert "Signal" not in src.__all__


def test_target_order_fill_keep_traceability_ids() -> None:
    from src import Fill, FillStatus, Order, Side, TargetPortfolio

    timestamp = datetime(2026, 1, 5)
    target = TargetPortfolio(
        target_id="TGT-1",
        strategy_id="demo",
        timestamp=timestamp,
        weights={"000001.SZ": Decimal("1")},
    )
    order = Order(
        order_id="ORD-1",
        target_id=target.target_id,
        strategy_id=target.strategy_id,
        symbol="000001.SZ",
        timestamp=timestamp,
        side=Side.BUY,
        quantity=100,
    )
    fill = Fill(
        fill_id="FIL-1",
        order_id=order.order_id,
        target_id=order.target_id,
        strategy_id=order.strategy_id,
        status=FillStatus.FILLED,
        symbol=order.symbol,
        timestamp=datetime(2026, 1, 6),
        side=order.side,
        quantity=order.quantity,
        price=Decimal("12.50"),
    )

    assert fill.target_id == target.target_id
    assert fill.strategy_id == target.strategy_id
