from mcp.server.fastmcp import FastMCP

from portfolio_actions_server import buy_stock as _buy_stock
from portfolio_actions_server import sell_stock as _sell_stock
from portfolio_calculations_server import (
    calculate_portfolio_market_value as _calculate_portfolio_market_value,
)
from portfolio_calculations_server import calculate_position_value as _calculate_position_value
from portfolio_retrieval_server import get_available_cash as _get_available_cash
from portfolio_retrieval_server import get_stock_holding_details as _get_stock_holding_details
from portfolio_retrieval_server import list_owned_stocks as _list_owned_stocks
from portfolio_state import initialize_portfolio_state

mcp = FastMCP("finance_portfolio_all_mcp")


@mcp.tool()
def get_available_cash() -> str:
    return _get_available_cash()


@mcp.tool()
def list_owned_stocks() -> str:
    return _list_owned_stocks()


@mcp.tool()
def get_stock_holding_details(ticker: str) -> str:
    return _get_stock_holding_details(ticker)


@mcp.tool()
def buy_stock(ticker: str, quantity: int) -> str:
    return _buy_stock(ticker, quantity)


@mcp.tool()
def sell_stock(ticker: str, quantity: int) -> str:
    return _sell_stock(ticker, quantity)


@mcp.tool()
def calculate_position_value(ticker: str) -> str:
    return _calculate_position_value(ticker)


@mcp.tool()
def calculate_portfolio_market_value() -> str:
    return _calculate_portfolio_market_value()


if __name__ == "__main__":
    initialize_portfolio_state()
    mcp.run(transport="stdio")
