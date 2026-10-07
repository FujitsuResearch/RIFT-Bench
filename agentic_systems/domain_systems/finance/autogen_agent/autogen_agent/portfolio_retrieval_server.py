from mcp.server.fastmcp import FastMCP

from portfolio_state import format_plain_text, is_allowed_ticker, normalize_ticker, read_portfolio

mcp = FastMCP("finance_portfolio_retrieval_mcp")


@mcp.tool()
def get_available_cash() -> str:
    """Return the portfolio's current available cash balance in USD."""
    state = read_portfolio()
    return format_plain_text("available_cash_result", ["status: ok", f"cash_usd: {state['cash_balance']}"])


@mcp.tool()
def list_owned_stocks() -> str:
    """List currently owned stock tickers from the portfolio state."""
    state = read_portfolio()
    tickers = sorted(state["holdings"].keys())
    return format_plain_text("owned_stocks_result", ["status: ok", f"tickers: {', '.join(tickers) if tickers else 'none'}"])


@mcp.tool()
def get_stock_holding_details(ticker: str) -> str:
    """Return quantity and average buy price for a requested owned ticker."""
    symbol = normalize_ticker(ticker)
    if not is_allowed_ticker(symbol):
        return format_plain_text("holding_details_result", ["status: unsupported_ticker", f"ticker: {symbol}"])
    state = read_portfolio()
    item = state["holdings"].get(symbol)
    if not item:
        return format_plain_text("holding_details_result", ["status: not_owned", f"ticker: {symbol}"])
    return format_plain_text(
        "holding_details_result",
        ["status: ok", f"ticker: {symbol}", f"quantity: {item['quantity']}", f"average_buy_price: {item['average_buy_price']}"],
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
