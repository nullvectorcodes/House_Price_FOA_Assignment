"""
Interactive CLI for Real-World House Price Prediction.
Allows users to input property attributes via command-line prompts and outputs
estimated price, price per sqft, confidence band, and feature contributions.
"""

from pathlib import Path
import sys
import joblib

from src.prediction import (
    VALID_FURNISHING,
    VALID_LOCATIONS,
    predict_house,
)


MODEL_PATH = Path("models/house_price_model.pkl")


def prompt_float(prompt: str, min_val: float, max_val: float, default: float) -> float:
    while True:
        try:
            val_str = input(f"{prompt} [default: {default}]: ").strip()
            if not val_str:
                return default
            val = float(val_str)
            if min_val <= val <= max_val:
                return val
            print(f"  ❌ Error: Value must be between {min_val} and {max_val}. Try again.")
        except ValueError:
            print("  ❌ Error: Please enter a valid numerical value.")


def prompt_int(prompt: str, min_val: int, max_val: int, default: int) -> int:
    while True:
        try:
            val_str = input(f"{prompt} [default: {default}]: ").strip()
            if not val_str:
                return default
            val = int(val_str)
            if min_val <= val <= max_val:
                return val
            print(f"  ❌ Error: Value must be between {min_val} and {max_val}. Try again.")
        except ValueError:
            print("  ❌ Error: Please enter a valid integer.")


def prompt_choice(prompt: str, options: list[str], default: str) -> str:
    options_str = "/".join(options)
    while True:
        val_str = input(f"{prompt} ({options_str}) [default: {default}]: ").strip()
        if not val_str:
            return default
        for opt in options:
            if opt.lower() == val_str.lower():
                return opt
        print(f"  ❌ Error: Please choose one of: {options_str}.")


def main() -> None:
    print("=" * 65)
    print(" 🏠 HOUSE PRICE PREDICTION SYSTEM - CLI INFERENCE")
    print("   Powered by Custom Batch Gradient Descent & Regularization")
    print("=" * 65)

    if not MODEL_PATH.exists():
        print(f"\n❌ Error: Model bundle not found at '{MODEL_PATH}'.")
        print("Please train the model first by running:")
        print("    python train.py\n")
        sys.exit(1)

    try:
        bundle = joblib.load(MODEL_PATH)
    except Exception as e:
        print(f"\n❌ Error loading model bundle: {e}")
        sys.exit(1)

    print("\nPlease enter property specifications (press Enter to accept default):\n")

    area_sqft = prompt_float("1.  Area in square feet", 100.0, 15000.0, 1800.0)
    bedrooms = prompt_int("2.  Number of bedrooms", 1, 10, 3)
    bathrooms = prompt_int("3.  Number of bathrooms", 1, 8, 2)
    stories = prompt_int("4.  Number of stories", 1, 6, 2)
    parking = prompt_int("5.  Parking spaces", 0, 6, 1)
    age_years = prompt_float("6.  Property age (years)", 0.0, 100.0, 5.0)

    location = prompt_choice("7.  Location", VALID_LOCATIONS, "Hyderabad")
    furnishing = prompt_choice("8.  Furnishing", VALID_FURNISHING, "Semi-Furnished")
    has_garden = prompt_choice("9.  Has Garden?", ["Yes", "No"], "Yes")
    has_pool = prompt_choice("10. Has Swimming Pool?", ["Yes", "No"], "No")

    distance_to_city_km = prompt_float("11. Distance to city center (km)", 0.0, 80.0, 8.0)
    distance_to_school_km = prompt_float("12. Distance to nearest school (km)", 0.0, 40.0, 2.0)
    distance_to_hospital_km = prompt_float("13. Distance to nearest hospital (km)", 0.0, 40.0, 3.0)

    crime_rate = prompt_float("14. Local crime rate (0.0 to 1.0)", 0.0, 1.0, 0.20)
    property_tax = prompt_float("15. Annual property tax (₹)", 0.0, 500000.0, 25000.0)
    income_index = prompt_float("16. Local income index (0.0 to 1.0)", 0.0, 1.0, 0.75)

    input_data = {
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "age_years": age_years,
        "distance_to_city_km": distance_to_city_km,
        "distance_to_school_km": distance_to_school_km,
        "distance_to_hospital_km": distance_to_hospital_km,
        "crime_rate": crime_rate,
        "property_tax": property_tax,
        "income_index": income_index,
        "location": location,
        "furnishing": furnishing,
        "has_garden": has_garden,
        "has_pool": has_pool,
    }

    try:
        res = predict_house(input_data, bundle)
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}\n")
        return

    print("\n" + "=" * 65)
    print(" 📊 VALUATION RESULT")
    print("=" * 65)
    print(f"  Estimated House Price:      {res['formatted_price']}")
    print(f"  Estimated Price / Sq.Ft:    {res['formatted_price_per_sqft']}")
    print(f"  Confidence Interval (95%):  {res['formatted_lower_bound']}  to  {res['formatted_upper_bound']}")
    print(f"  Model Error Band (RMSE):    ± {res['formatted_rmse']}")
    print("-" * 65)
    print("  Key Model-Based Contributing Factors:")
    for item in res["explanations"]:
        icon = "📈" if item["is_positive"] else "📉"
        print(f"    {icon} {item['explanation']}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
