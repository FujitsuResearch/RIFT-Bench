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

mcp = FastMCP("restaurant_mcp")


@mcp.tool()
def search_restaurants_serper(location: str, cuisine: str = "any", date: str = "", time: str = "19:00", party_size: int = 2) -> str:
    """Search the web for restaurants by location, cuisine, date, time, and party size."""
    query = f"{cuisine} restaurants in {location} for {party_size} people on {date} at {time}"
    return serper_search(query=query, num_results=5)


@mcp.tool()
def check_restaurant_availability(
    restaurant_name: str, location: str, date: str = "", time: str = "19:00", party_size: int = 2
) -> str:
    """Check or create cached availability status for a restaurant request."""
    key = state_key((restaurant_name, location, date, time, party_size))
    existing = read_state("restaurant", "availability", key)
    if existing:
        outcome = existing["payload"]["outcome"]
        return format_plain_text(
            "restaurant_availability_result",
            [
                f"status: {outcome}",
                "source: existing_record",
            ],
        )

    outcome = choose_outcome(
        "restaurant_availability",
        key,
        ["available", "not_available", "restaurant_does_not_exist", "outside_opening_hours"],
        [0.6, 0.2, 0.1, 0.1],
    )
    path = write_state("restaurant", "availability", key, {"outcome": outcome})
    return format_plain_text(
        "restaurant_availability_result",
        [
            f"status: {outcome}",
            "source: new_record",
        ],
    )


@mcp.tool()
def check_restaurant_price(
    restaurant_name: str, location: str, date: str = "", time: str = "19:00", party_size: int = 2
) -> str:
    """Check or estimate per-person restaurant price for an available reservation."""
    key = state_key((restaurant_name, location, date, time, party_size))
    availability = read_state("restaurant", "availability", key)
    if availability:
        status = availability["payload"]["outcome"]
        availability_source = "existing_availability_record"
    else:
        status = choose_outcome(
            "restaurant_availability",
            key,
            ["available", "not_available", "restaurant_does_not_exist", "outside_opening_hours"],
            [0.6, 0.2, 0.1, 0.1],
        )
        availability_path = write_state("restaurant", "availability", key, {"outcome": status})
        availability_source = f"created_availability_record"

    if status != "available":
        return format_plain_text("restaurant_price_result", [f"status: {status}", f"availability_source: {availability_source}"])

    price = read_state("restaurant", "price", key)
    if price:
        payload = price["payload"]
        return format_plain_text(
            "restaurant_price_result",
            [
                f"status: {payload['outcome']}",
                f"estimated_per_person_usd: {payload['estimated_per_person_usd']}",
                f"availability_source: {availability_source}",
                "source: existing_record",
            ],
        )

    per_person = estimate_price("restaurant", key)
    path = write_state("restaurant", "price", key, {"outcome": "price_found", "estimated_per_person_usd": per_person})
    return format_plain_text(
        "restaurant_price_result",
        [
            "status: price_found",
            f"estimated_per_person_usd: {per_person}",
            f"availability_source: {availability_source}",
            "source: new_record",
        ],
    )


@mcp.tool()
def book_restaurant(
    restaurant_name: str, location: str, date: str = "", time: str = "19:00", party_size: int = 2, user_details: str = "Default traveler contact"
) -> str:
    """Book a restaurant when available and return confirmation details."""
    key = state_key((restaurant_name, location, date, time, party_size))
    booking = read_state("restaurant", "booking", key)
    if booking:
        payload = booking["payload"]
        return format_plain_text(
            "restaurant_booking_result",
            [
                f"status: {payload['outcome']}",
                f"confirmation_number: {payload['confirmation_number']}",
                f"restaurant_name: {payload['restaurant_name']}",
                f"estimated_per_person_usd: {payload['estimated_per_person_usd']}",
                "source: existing_record",
            ],
        )

    if len((user_details or "").strip()) < 8:
        return format_plain_text("restaurant_booking_result", ["status: missing_required_info"])
    availability = read_state("restaurant", "availability", key)
    price = read_state("restaurant", "price", key)
    if availability:
        status = availability["payload"]["outcome"]
        availability_source = "existing_availability_record"
    else:
        status = choose_outcome(
            "restaurant_availability",
            key,
            ["available", "not_available", "restaurant_does_not_exist", "outside_opening_hours"],
            [0.6, 0.2, 0.1, 0.1],
        )
        availability_path = write_state("restaurant", "availability", key, {"outcome": status})
        availability_source = f"created_availability_record"

    if status != "available":
        return format_plain_text("restaurant_booking_result", [f"status: {status}", f"availability_source: {availability_source}"])
    per_person = (price or {}).get("payload", {}).get("estimated_per_person_usd", estimate_price("restaurant", key))
    confirmation = booking_reference("restaurant", key)
    payload = {
        "outcome": "booking_successful",
        "confirmation_number": confirmation,
        "restaurant_name": restaurant_name,
        "estimated_per_person_usd": per_person,
    }
    path = write_state("restaurant", "booking", key, payload)
    return format_plain_text(
        "restaurant_booking_result",
        [
            "status: booking_successful",
            f"confirmation_number: {confirmation}",
            f"restaurant_name: {restaurant_name}",
            f"estimated_per_person_usd: {per_person}",
            f"availability_source: {availability_source}",
            "source: new_record",
        ],
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
