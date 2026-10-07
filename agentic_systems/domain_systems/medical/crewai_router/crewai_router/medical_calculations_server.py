from datetime import date

from mcp.server.fastmcp import FastMCP

mcp = FastMCP("medical_calculations_mcp")


def _to_kg(weight: float, weight_unit: str) -> float:
    unit = weight_unit.lower()
    if unit == "kg":
        return weight
    if unit == "lb":
        return weight * 0.45359237
    return 0


def _to_m(height: float, height_unit: str) -> float:
    unit = height_unit.lower()
    if unit == "m":
        return height
    if unit == "cm":
        return height / 100.0
    if unit == "in":
        return height * 0.0254
    return 0


@mcp.tool()
def calculate_bmi(height: float, height_unit: str, weight: float, weight_unit: str) -> str:
    """Calculate BMI from height and weight and return category with numeric result."""
    h_m = _to_m(height, height_unit)
    w_kg = _to_kg(weight, weight_unit)
    if h_m <= 0:
        return "type: calculate_bmi_result\nstatus: invalid_height"
    bmi = w_kg / (h_m * h_m)
    if bmi < 18.5:
        category = "underweight"
    elif bmi < 25:
        category = "normal"
    elif bmi < 30:
        category = "overweight"
    else:
        category = "obese"
    return "\n".join(
        [
            "type: calculate_bmi_result",
            "status: ok",
            f"bmi: {bmi:.2f}",
            f"category: {category}",
        ]
    )


@mcp.tool()
def convert_medical_units(value: float, from_unit: str, to_unit: str) -> str:
    """Convert between supported medical measurement units and return the converted value."""
    src = from_unit.lower()
    dst = to_unit.lower()
    factors = {
        ("mg", "mcg"): 1000.0,
        ("mcg", "mg"): 0.001,
        ("g", "mg"): 1000.0,
        ("mg", "g"): 0.001,
        ("kg", "lb"): 2.2046226218,
        ("lb", "kg"): 0.45359237,
        ("cm", "in"): 0.3937007874,
        ("in", "cm"): 2.54,
        ("ml", "l"): 0.001,
        ("l", "ml"): 1000.0,
    }
    if src == dst:
        converted = value
    elif (src, dst) in factors:
        converted = value * factors[(src, dst)]
    else:
        return "type: convert_medical_units_result\nstatus: unsupported_conversion"
    return "\n".join(
        [
            "type: convert_medical_units_result",
            "status: ok",
            f"input_value: {value}",
            f"input_unit: {from_unit}",
            f"output_value: {converted}",
            f"output_unit: {to_unit}",
        ]
    )


@mcp.tool()
def age_from_date(date_of_birth: str, reference_date: str = "") -> str:
    """Calculate age in years from date of birth and an optional reference date."""
    dob = date.fromisoformat(date_of_birth)
    ref = date.fromisoformat(reference_date) if reference_date else date.today()
    years = ref.year - dob.year - ((ref.month, ref.day) < (dob.month, dob.day))
    return "\n".join(
        [
            "type: age_from_date_result",
            "status: ok",
            f"date_of_birth: {dob.isoformat()}",
            f"reference_date: {ref.isoformat()}",
            f"age_years: {years}",
        ]
    )


if __name__ == "__main__":
    mcp.run(transport="stdio")
