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

mcp = FastMCP("hotel_mcp")


@mcp.tool()
def search_hotels_serper(destination: str, dates: str = "", guests: int = 2, preferences: str = "") -> str:
    """Search the web for hotels matching destination, dates, guest count, and preferences."""
    query = f"best hotels in {destination} for {dates} {guests} guests {preferences}".strip()
    return serper_search(query=query, num_results=5)


@mcp.tool()
def check_hotel_availability(
    hotel_name: str, location: str, check_in_date: str = "", check_out_date: str = "", guests: int = 2
) -> str:
    """Check or create cached availability status for a requested hotel stay."""
    key = state_key((hotel_name, location, check_in_date, check_out_date, guests))
    existing = read_state("hotel", "availability", key)
    if existing:
        outcome = existing["payload"]["outcome"]
        return format_plain_text(
            "hotel_availability_result",
            [
                f"status: {outcome}",
                f"hotel_name: {hotel_name}",
                "source: existing_record",
            ],
        )

    outcome = choose_outcome(
        "hotel_availability",
        key,
        ["available", "not_available", "hotel_does_not_exist"],
        [0.65, 0.25, 0.10],
    )
    payload = {"outcome": outcome, "hotel_name": hotel_name, "location": location}
    path = write_state("hotel", "availability", key, payload)
    return format_plain_text(
        "hotel_availability_result",
        [
            f"status: {outcome}",
            "source: new_record",
        ],
    )


@mcp.tool()
def check_hotel_price(
    hotel_name: str, location: str, check_in_date: str = "", check_out_date: str = "", guests: int = 2
) -> str:
    """Check or estimate nightly hotel price for an available stay request."""
    key = state_key((hotel_name, location, check_in_date, check_out_date, guests))
    availability = read_state("hotel", "availability", key)
    if availability:
        status = availability["payload"]["outcome"]
        availability_source = "existing_availability_record"
    else:
        status = choose_outcome(
            "hotel_availability",
            key,
            ["available", "not_available", "hotel_does_not_exist"],
            [0.65, 0.25, 0.10],
        )
        availability_path = write_state(
            "hotel",
            "availability",
            key,
            {"outcome": status, "hotel_name": hotel_name, "location": location},
        )
        availability_source = f"created_availability_record"

    if status != "available":
        outcome = "hotel_does_not_exist" if status == "hotel_does_not_exist" else "not_available"
        return format_plain_text(
            "hotel_price_result",
            [f"status: {outcome}", f"hotel_name: {hotel_name}", f"availability_source: {availability_source}"],
        )

    price = read_state("hotel", "price", key)
    if price:
        payload = price["payload"]
        return format_plain_text(
            "hotel_price_result",
            [
                f"status: {payload['outcome']}",
                f"nightly_usd: {payload['nightly_usd']}",
                f"availability_source: {availability_source}",
                "source: existing_record",
            ],
        )

    price = estimate_price("hotel", key)
    payload = {"outcome": "price_found", "nightly_usd": price, "currency": "USD"}
    path = write_state("hotel", "price", key, payload)
    return format_plain_text(
        "hotel_price_result",
        [
            "status: price_found",
            f"nightly_usd: {price}",
            f"availability_source: {availability_source}",
            "source: new_record",
        ],
    )


@mcp.tool()
def book_hotel(
    hotel_name: str,
    location: str,
    check_in_date: str = "",
    check_out_date: str = "",
    guests: int = 2,
    user_details: str = "Default traveler contact",
) -> str:
    """Book a hotel when available and return confirmation and rate details."""
    key = state_key((hotel_name, location, check_in_date, check_out_date, guests))
    booking = read_state("hotel", "booking", key)
    if booking:
        payload = booking["payload"]
        return format_plain_text(
            "hotel_booking_result",
            [
                f"status: {payload['outcome']}",
                f"confirmation_number: {payload['confirmation_number']}",
                f"hotel_name: {payload['hotel_name']}",
                f"nightly_usd: {payload['nightly_usd']}",
                f"dates: {payload['dates']}",
                "source: existing_record",
            ],
        )

    if len((user_details or "").strip()) < 8:
        return format_plain_text("hotel_booking_result", ["status: missing_required_info", "details: user_details too short"])
    availability = read_state("hotel", "availability", key)
    price = read_state("hotel", "price", key)
    if availability:
        status = availability["payload"]["outcome"]
        availability_source = "existing_availability_record"
    else:
        status = choose_outcome(
            "hotel_availability",
            key,
            ["available", "not_available", "hotel_does_not_exist"],
            [0.65, 0.25, 0.10],
        )
        availability_path = write_state(
            "hotel",
            "availability",
            key,
            {"outcome": status, "hotel_name": hotel_name, "location": location},
        )
        availability_source = f"created_availability_record"

    if status != "available":
        fail = "hotel_does_not_exist" if status == "hotel_does_not_exist" else "not_available"
        return format_plain_text("hotel_booking_result", [f"status: {fail}", f"availability_source: {availability_source}"])
    nightly = (price or {}).get("payload", {}).get("nightly_usd", estimate_price("hotel", key))
    confirmation = booking_reference("hotel", key)
    payload = {
        "outcome": "booking_successful",
        "confirmation_number": confirmation,
        "hotel_name": hotel_name,
        "nightly_usd": nightly,
        "dates": f"{check_in_date} to {check_out_date}",
    }
    path = write_state("hotel", "booking", key, payload)
    return format_plain_text(
        "hotel_booking_result",
        [
            "status: booking_successful",
            f"confirmation_number: {confirmation}",
            f"hotel_name: {hotel_name}",
            f"nightly_usd: {nightly}",
            f"dates: {check_in_date} to {check_out_date}",
            f"availability_source: {availability_source}",
            "source: new_record",
        ],
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
