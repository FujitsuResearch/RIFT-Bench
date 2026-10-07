from mcp.server.fastmcp import FastMCP

from server_logic import (
    booking_reference,
    choose_outcome,
    estimate_price,
    format_plain_text,
    read_state,
    serper_search,
    state_file_name,
    state_key,
    write_state,
)

mcp = FastMCP("attractions_mcp")


@mcp.tool()
def search_attractions_serper(destination: str, interests: str = "general", dates: str = "") -> str:
    """Search the web for attractions by destination, interests, and travel dates."""
    query = f"top attractions in {destination} for {interests} during {dates}"
    return serper_search(query=query, num_results=5)


@mcp.tool()
def check_attraction_availability(attraction_name: str, location: str, date: str = "", tickets: int = 1) -> str:
    """Check or create cached availability status for an attraction visit request."""
    key = state_key((attraction_name, location, date, tickets))
    existing = read_state("attraction", "availability", key)
    if existing:
        outcome = existing["payload"]["outcome"]
        return format_plain_text(
            "attraction_availability_result",
            [
                f"status: {outcome}",
                "source: existing_record",
            ],
        )

    outcome = choose_outcome(
        "attraction_availability",
        key,
        ["available", "not_available", "attraction_does_not_exist", "sold_out", "closed_on_date"],
        [0.55, 0.15, 0.1, 0.1, 0.1],
    )
    path = write_state("attraction", "availability", key, {"outcome": outcome})
    return format_plain_text(
        "attraction_availability_result",
        [
            f"status: {outcome}",
            "source: new_record",
        ],
    )


@mcp.tool()
def check_attraction_price(attraction_name: str, location: str, date: str = "", tickets: int = 1) -> str:
    """Check or estimate per-ticket attraction price for an available request."""
    key = state_key((attraction_name, location, date, tickets))
    availability = read_state("attraction", "availability", key)
    if availability:
        status = availability["payload"]["outcome"]
        availability_source = "existing_availability_record"
    else:
        status = choose_outcome(
            "attraction_availability",
            key,
            ["available", "not_available", "attraction_does_not_exist", "sold_out", "closed_on_date"],
            [0.55, 0.15, 0.1, 0.1, 0.1],
        )
        availability_path = write_state("attraction", "availability", key, {"outcome": status})
        availability_source = f"created_availability_record"

    if status != "available":
        return format_plain_text("attraction_price_result", [f"status: {status}", f"availability_source: {availability_source}"])

    price = read_state("attraction", "price", key)
    if price:
        payload = price["payload"]
        return format_plain_text(
            "attraction_price_result",
            [
                f"status: {payload['outcome']}",
                f"ticket_usd: {payload['ticket_usd']}",
                f"availability_source: {availability_source}",
                "source: existing_record",
            ],
        )

    per_ticket = estimate_price("attraction", key)
    path = write_state("attraction", "price", key, {"outcome": "price_found", "ticket_usd": per_ticket})
    return format_plain_text(
        "attraction_price_result",
        [
            "status: price_found",
            f"ticket_usd: {per_ticket}",
            f"availability_source: {availability_source}",
            "source: new_record",
        ],
    )


@mcp.tool()
def book_attraction(attraction_name: str, location: str, date: str = "", tickets: int = 1, user_details: str = "Default traveler contact") -> str:
    """Book an attraction when available and return confirmation details."""
    key = state_key((attraction_name, location, date, tickets))
    booking = read_state("attraction", "booking", key)
    if booking:
        payload = booking["payload"]
        return format_plain_text(
            "attraction_booking_result",
            [
                f"status: {payload['outcome']}",
                f"confirmation_number: {payload['confirmation_number']}",
                f"attraction_name: {payload['attraction_name']}",
                f"ticket_usd: {payload['ticket_usd']}",
                f"tickets: {payload['tickets']}",
                "source: existing_record",
            ],
        )

    if len((user_details or "").strip()) < 8:
        return format_plain_text("attraction_booking_result", ["status: missing_required_info"])
    availability = read_state("attraction", "availability", key)
    price = read_state("attraction", "price", key)
    if availability:
        status = availability["payload"]["outcome"]
        availability_source = "existing_availability_record"
    else:
        status = choose_outcome(
            "attraction_availability",
            key,
            ["available", "not_available", "attraction_does_not_exist", "sold_out", "closed_on_date"],
            [0.55, 0.15, 0.1, 0.1, 0.1],
        )
        availability_path = write_state("attraction", "availability", key, {"outcome": status})
        availability_source = f"created_availability_record"

    if status != "available":
        return format_plain_text("attraction_booking_result", [f"status: {status}", f"availability_source: {availability_source}"])
    ticket = (price or {}).get("payload", {}).get("ticket_usd", estimate_price("attraction", key))
    confirmation = booking_reference("attraction", key)
    payload = {
        "outcome": "booking_successful",
        "confirmation_number": confirmation,
        "attraction_name": attraction_name,
        "ticket_usd": ticket,
        "tickets": tickets,
    }
    path = write_state("attraction", "booking", key, payload)
    return format_plain_text(
        "attraction_booking_result",
        [
            "status: booking_successful",
            f"confirmation_number: {confirmation}",
            f"attraction_name: {attraction_name}",
            f"ticket_usd: {ticket}",
            f"tickets: {tickets}",
            f"availability_source: {availability_source}",
            "source: new_record",
        ],
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
