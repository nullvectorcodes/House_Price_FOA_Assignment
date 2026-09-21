"""
Interactive CLI for Real-World House Price Prediction with Location Intelligence.
Automatically geocodes property locations, retrieves nearby POIs via Haversine distance,
and computes estimated property valuation.
"""

from pathlib import Path
import sys
import joblib

from services.location_features import extract_location_features
from src.prediction import (
    VALID_FURNISHING,
    predict_house,
)


MODEL_PATH = Path("models/house_price_model.pkl")


def prompt_str(prompt: str, default: str) -> str:
    val_str = input(f"{prompt} [default: {default}]: ").strip()
    return val_str if val_str else default


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
    print("=" * 70)
    print(" 🏠 HOUSE PRICE PREDICTION SYSTEM - CLI INFERENCE")
    print("   Location-Aware Automatic Feature Engineering & POI Intelligence")
    print("=" * 70)

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

    location_query = prompt_str("1.  Property Location / Address", "Gachibowli, Hyderabad")
    area_sqft = prompt_float("2.  Area in square feet", 100.0, 15000.0, 1800.0)
    bedrooms = prompt_int("3.  Number of bedrooms (BHK)", 1, 10, 3)
    bathrooms = prompt_int("4.  Number of bathrooms", 1, 8, 2)
    stories = prompt_int("5.  Number of stories/floors", 1, 6, 2)
    parking = prompt_int("6.  Parking spaces", 0, 6, 1)
    age_years = prompt_float("7.  Property age (years)", 0.0, 100.0, 5.0)

    furnishing = prompt_choice("8.  Furnishing", VALID_FURNISHING, "Semi-Furnished")
    has_garden = prompt_choice("9.  Has Garden?", ["Yes", "No"], "Yes")
    has_pool = prompt_choice("10. Has Swimming Pool?", ["Yes", "No"], "No")

    print("\n🔍 Geocoding location and extracting nearby amenities...")
    try:
        loc_features = extract_location_features(location_query, radius_km=2.0)
        print(f"  ✓ Geocoded to: {loc_features.get('formatted_address', location_query)}")
        print(f"  ✓ Coordinates: {loc_features['latitude']}° N, {loc_features['longitude']}° E")
        print(f"  ✓ Distance to City Center (Haversine): {loc_features['distance_to_city_center']} km")
        print(f"  ✓ Schools Nearby (2km): {loc_features['schools_within_2km']} (nearest: {loc_features['nearest_school_km']} km)")
        print(f"  ✓ Hospitals Nearby (2km): {loc_features['hospitals_within_2km']} (nearest: {loc_features['nearest_hospital_km']} km)")
        print(f"  ✓ Metro Nearby (2km): {loc_features['metro_within_2km']} (nearest: {loc_features['nearest_metro_km']} km)")
    except Exception as e:
        print(f"  ⚠️ Geocoding notice: {e}")
        loc_features = {"location": "Hyderabad"}

    input_data = {
        "location": location_query,
        "area_sqft": area_sqft,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "stories": stories,
        "parking": parking,
        "age_years": age_years,
        "furnishing": furnishing,
        "has_garden": has_garden,
        "has_pool": has_pool,
        **loc_features,
    }

    try:
        res = predict_house(input_data, bundle)
    except ValueError as e:
        print(f"\n❌ Validation Error: {e}\n")
        return

    print("\n" + "=" * 70)
    print(" 📊 VALUATION RESULT")
    print("=" * 70)
    print(f"  Estimated House Price:      {res['formatted_price']}")
    print(f"  Estimated Price / Sq.Ft:    {res['formatted_price_per_sqft']}")
    print(f"  Confidence Interval (95%):  {res['formatted_lower_bound']}  to  {res['formatted_upper_bound']}")
    print(f"  Model Error Band (RMSE):    ± {res['formatted_rmse']}")
    print("-" * 70)
    print("  Key Model-Based Contributing Factors:")
    for item in res["explanations"]:
        icon = "📈" if item["is_positive"] else "📉"
        print(f"    {icon} {item['explanation']}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
