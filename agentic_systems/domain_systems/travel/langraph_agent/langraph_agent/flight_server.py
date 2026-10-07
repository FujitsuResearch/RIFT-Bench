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

mcp = FastMCP("flight_mcp")


@mcp.tool()
def search_flights_serper(
    origin: str, destination: str = "", departure_date: str = "", return_date: str = "", passengers: int = 1
) -> str:
    """Search the web for flight options matching route, dates, and passenger count."""
    query = f"flights {origin} to {destination} departing {departure_date} return {return_date} for {passengers} passengers"
    return serper_search(query=query, num_results=5)


@mcp.tool()
def check_flight_availability(
    origin: str, destination: str, departure_date: str = "", return_date: str = "", passengers: int = 1
) -> str:
    """Check or create cached availability status for a requested flight itinerary."""
    key = state_key((origin, destination, departure_date, return_date, passengers))
    existing = read_state("flight", "availability", key)
    if existing:
        outcome = existing["payload"]["outcome"]
        return format_plain_text(
            "flight_availability_result",
            [
                f"status: {outcome}",
                "source: existing_record",
            ],
        )

    outcome = choose_outcome(
        "flight_availability",
        key,
        ["available", "not_available", "route_not_found", "flight_does_not_exist"],
        [0.62, 0.2, 0.1, 0.08],
    )
    payload = {"outcome": outcome, "origin": origin, "destination": destination}
    path = write_state("flight", "availability", key, payload)
    return format_plain_text(
        "flight_availability_result",
        [
            f"status: {outcome}",
            "source: new_record",
        ],
    )


@mcp.tool()
def check_flight_price(
    origin: str, destination: str, departure_date: str = "", return_date: str = "", passengers: int = 1
) -> str:
    """Check or estimate ticket price for an available flight itinerary."""
    key = state_key((origin, destination, departure_date, return_date, passengers))
    availability = read_state("flight", "availability", key)
    if availability:
        status = availability["payload"]["outcome"]
        availability_source = "existing_availability_record"
    else:
        status = choose_outcome(
            "flight_availability",
            key,
            ["available", "not_available", "route_not_found", "flight_does_not_exist"],
            [0.62, 0.2, 0.1, 0.08],
        )
        availability_path = write_state(
            "flight",
            "availability",
            key,
            {"outcome": status, "origin": origin, "destination": destination},
        )
        availability_source = f"created_availability_record"

    if status != "available":
        return format_plain_text("flight_price_result", [f"status: {status}", f"availability_source: {availability_source}"])

    price = read_state("flight", "price", key)
    if price:
        payload = price["payload"]
        return format_plain_text(
            "flight_price_result",
            [
                f"status: {payload['outcome']}",
                f"ticket_usd: {payload['ticket_usd']}",
                f"availability_source: {availability_source}",
                "source: existing_record",
            ],
        )

    price = estimate_price("flight", key)
    payload = {"outcome": "price_found", "ticket_usd": price, "currency": "USD"}
    path = write_state("flight", "price", key, payload)
    return format_plain_text(
        "flight_price_result",
        [
            "status: price_found",
            f"ticket_usd: {price}",
            f"availability_source: {availability_source}",
            "source: new_record",
        ],
    )


@mcp.tool()
def book_flight(
    origin: str,
    destination: str,
    departure_date: str = "",
    return_date: str = "",
    passengers: int = 1,
    user_details: str = "Default traveler contact",
) -> str:
    """Book a flight when available and return confirmation details."""
    key = state_key((origin, destination, departure_date, return_date, passengers))
    booking = read_state("flight", "booking", key)
    if booking:
        payload = booking["payload"]
        return format_plain_text(
            "flight_booking_result",
            [
                f"status: {payload['outcome']}",
                f"confirmation_number: {payload['confirmation_number']}",
                f"route: {payload['route']}",
                f"ticket_usd: {payload['ticket_usd']}",
                "source: existing_record",
            ],
        )

    if len((user_details or "").strip()) < 8:
        return format_plain_text("flight_booking_result", ["status: missing_required_info"])
    availability = read_state("flight", "availability", key)
    price = read_state("flight", "price", key)
    if availability:
        status = availability["payload"]["outcome"]
        availability_source = "existing_availability_record"
    else:
        status = choose_outcome(
            "flight_availability",
            key,
            ["available", "not_available", "route_not_found", "flight_does_not_exist"],
            [0.62, 0.2, 0.1, 0.08],
        )
        availability_path = write_state(
            "flight",
            "availability",
            key,
            {"outcome": status, "origin": origin, "destination": destination},
        )
        availability_source = f"created_availability_record"

    if status != "available":
        return format_plain_text("flight_booking_result", [f"status: {status}", f"availability_source: {availability_source}"])
    ticket = (price or {}).get("payload", {}).get("ticket_usd", estimate_price("flight", key))
    confirmation = booking_reference("flight", key)
    payload = {
        "outcome": "booking_successful",
        "confirmation_number": confirmation,
        "route": f"{origin} -> {destination}",
        "ticket_usd": ticket,
    }
    path = write_state("flight", "booking", key, payload)
    return format_plain_text(
        "flight_booking_result",
        [
            "status: booking_successful",
            f"confirmation_number: {confirmation}",
            f"route: {origin} -> {destination}",
            f"ticket_usd: {ticket}",
            f"availability_source: {availability_source}",
            "source: new_record",
        ],
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
