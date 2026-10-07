from mcp.server.fastmcp import FastMCP

from portfolio_state import deterministic_price, format_plain_text, is_allowed_ticker, normalize_ticker, read_portfolio, write_portfolio

mcp = FastMCP("finance_portfolio_actions_mcp")


@mcp.tool()
def buy_stock(ticker: str, quantity: int) -> str:
    """Execute a buy order for a supported ticker and update portfolio cash/holdings."""
    symbol = normalize_ticker(ticker)
    if not is_allowed_ticker(symbol):
        return format_plain_text("buy_result", ["status: unsupported_ticker", f"ticker: {symbol}"])
    if quantity <= 0:
        return format_plain_text("buy_result", ["status: invalid_quantity"])
    state = read_portfolio()
    px = deterministic_price(symbol)
    cost = round(px * quantity, 2)
    if state["cash_balance"] < cost:
        return format_plain_text("buy_result", ["status: insufficient_cash", f"required_usd: {cost}"])
    item = state["holdings"].get(symbol, {"quantity": 0, "average_buy_price": 0.0})
    old_q = item["quantity"]
    new_q = old_q + quantity
    new_avg = round(((item["average_buy_price"] * old_q) + cost) / new_q, 2) if new_q else 0.0
    state["holdings"][symbol] = {"quantity": new_q, "average_buy_price": new_avg}
    state["cash_balance"] = round(state["cash_balance"] - cost, 2)
    state["order_history"].append({"action": "buy", "ticker": symbol, "quantity": quantity, "price": px})
    write_portfolio(state)
    return format_plain_text(
        "buy_result",
        ["status: executed", f"ticker: {symbol}", f"quantity: {quantity}", f"price_usd: {px}", f"cash_after_usd: {state['cash_balance']}"],
    )


@mcp.tool()
def sell_stock(ticker: str, quantity: int) -> str:
    """Execute a sell order for owned shares and update portfolio cash/holdings."""
    symbol = normalize_ticker(ticker)
    if not is_allowed_ticker(symbol):
        return format_plain_text("sell_result", ["status: unsupported_ticker", f"ticker: {symbol}"])
    if quantity <= 0:
        return format_plain_text("sell_result", ["status: invalid_quantity"])
    state = read_portfolio()
    item = state["holdings"].get(symbol)
    if not item or item["quantity"] <= 0:
        return format_plain_text("sell_result", ["status: not_owned", f"ticker: {symbol}"])
    if item["quantity"] < quantity:
        return format_plain_text("sell_result", ["status: insufficient_shares", f"available_quantity: {item['quantity']}"])
    px = deterministic_price(symbol)
    proceeds = round(px * quantity, 2)
    remaining = item["quantity"] - quantity
    if remaining == 0:
        del state["holdings"][symbol]
    else:
        state["holdings"][symbol]["quantity"] = remaining
    state["cash_balance"] = round(state["cash_balance"] + proceeds, 2)
    state["order_history"].append({"action": "sell", "ticker": symbol, "quantity": quantity, "price": px})
    write_portfolio(state)
    return format_plain_text(
        "sell_result",
        ["status: executed", f"ticker: {symbol}", f"quantity: {quantity}", f"price_usd: {px}", f"cash_after_usd: {state['cash_balance']}"],
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
