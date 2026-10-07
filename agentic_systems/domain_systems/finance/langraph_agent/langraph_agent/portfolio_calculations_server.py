from mcp.server.fastmcp import FastMCP

from portfolio_state import deterministic_price, format_plain_text, is_allowed_ticker, normalize_ticker, read_portfolio

mcp = FastMCP("finance_portfolio_calculations_mcp")


@mcp.tool()
def calculate_position_value(ticker: str) -> str:
    """Calculate current USD market value for one owned ticker position."""
    symbol = normalize_ticker(ticker)
    if not is_allowed_ticker(symbol):
        return format_plain_text("position_value_result", ["status: unsupported_ticker", f"ticker: {symbol}"])
    state = read_portfolio()
    item = state["holdings"].get(symbol)
    if not item:
        return format_plain_text("position_value_result", ["status: not_owned", f"ticker: {symbol}"])
    px = deterministic_price(symbol)
    value = round(item["quantity"] * px, 2)
    return format_plain_text(
        "position_value_result",
        ["status: ok", f"ticker: {symbol}", f"current_price_usd: {px}", f"market_value_usd: {value}"],
    )


@mcp.tool()
def calculate_portfolio_market_value() -> str:
    """Calculate total USD market value across all current portfolio holdings."""
    state = read_portfolio()
    total = 0.0
    for symbol, item in state["holdings"].items():
        total += deterministic_price(symbol) * item["quantity"]
    total = round(total, 2)
    return format_plain_text("portfolio_market_value_result", ["status: ok", f"market_value_usd: {total}"])


if __name__ == "__main__":
    mcp.run(transport="stdio")
