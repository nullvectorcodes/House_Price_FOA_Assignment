"""
Inference and Real-World House Price Prediction Module.
Validates realistic constraints, applies automatic location geocoding & POI feature enrichment,
and calculates price, price/sqft, confidence bands, and feature explanations.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from services.location_features import extract_location_features
from src.evaluation import compute_confidence_band, format_inr


VALID_LOCATIONS = ["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Pune", "Chennai"]
VALID_FURNISHING = ["Unfurnished", "Semi-Furnished", "Furnished"]


def validate_property_input(data: Dict[str, Any]) -> None:
    """Validates property input constraints."""
    enrich_and_validate_input(data)


def enrich_and_validate_input(data: Dict[str, Any]) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]]]:
    """
    Enriches user input with automatic location features and validates constraints.

    Returns
    -------
    Tuple[Dict[str, Any], Optional[Dict[str, Any]]]
        (enriched_data_dict, location_intelligence_dict)
    """
    data_out = dict(data)
    location_str = str(data_out.get("location", "")).strip()
    if not location_str:
        raise ValueError("Property location is required. Please enter an address, locality, or city.")

    location_intel = None

    # Check if geographic features are missing; if so, extract automatically
    needs_geo = (
        "latitude" not in data_out
        or "longitude" not in data_out
        or "distance_to_city_km" not in data_out
        or "schools_within_2km" not in data_out
    )

    if needs_geo:
        try:
            geo_features = extract_location_features(location_str)
            location_intel = geo_features
            for k, v in geo_features.items():
                if k not in data_out:
                    data_out[k] = v
        except Exception:
            # Fallback values if live and offline geocoding fail
            data_out.setdefault("latitude", 17.3850)
            data_out.setdefault("longitude", 78.4867)
            data_out.setdefault("distance_to_city_km", 8.0)
            data_out.setdefault("distance_to_school_km", 1.2)
            data_out.setdefault("distance_to_hospital_km", 1.8)
            data_out.setdefault("schools_within_2km", 8)
            data_out.setdefault("nearest_school_km", 1.2)
            data_out.setdefault("hospitals_within_2km", 4)
            data_out.setdefault("nearest_hospital_km", 1.8)
            data_out.setdefault("metro_within_2km", 2)
            data_out.setdefault("nearest_metro_km", 1.5)
            data_out.setdefault("parks_within_2km", 3)
            data_out.setdefault("shopping_within_2km", 2)
            data_out.setdefault("supermarkets_within_2km", 5)

    # Always resolve location to one of the trained metro cities
    matched_city = None
    for metro in VALID_LOCATIONS:
        if metro.lower() in location_str.lower():
            matched_city = metro
            break
    if not matched_city and location_intel:
        intel_city = str(location_intel.get("city", "")).title()
        for metro in VALID_LOCATIONS:
            if metro.lower() in intel_city.lower():
                matched_city = metro
                break
    data_out["location"] = matched_city or "Hyderabad"

    # Defaults for optional/secondary properties
    data_out.setdefault("crime_rate", 0.18)
    data_out.setdefault("property_tax", 25000.0)
    data_out.setdefault("income_index", 0.75)
    data_out.setdefault("has_garden", "No")
    data_out.setdefault("has_pool", "No")
    data_out.setdefault("stories", 2)
    data_out.setdefault("parking", 1)

    # Numerical validation rules
    numeric_checks = [
        ("area_sqft", 100.0, 20000.0, "Area (sq.ft) must be a positive number greater than 100."),
        ("bedrooms", 1, 20, "Bedrooms must be an integer >= 1."),
        ("bathrooms", 1, 15, "Bathrooms must be an integer >= 1."),
        ("stories", 1, 10, "Stories must be an integer >= 1."),
        ("parking", 0, 10, "Parking spaces must be a non-negative integer (>= 0)."),
        ("age_years", 0.0, 150.0, "Property age must be a non-negative number (>= 0)."),
        ("distance_to_city_km", 0.0, 200.0, "Distance to city center must be >= 0 km."),
        ("crime_rate", 0.0, 1.0, "Crime rate must be between 0.0 and 1.0."),
        ("property_tax", 0.0, 1000000.0, "Property tax must be >= 0."),
        ("income_index", 0.0, 1.0, "Income index must be between 0.0 and 1.0."),
    ]

    for field, min_val, max_val, err_msg in numeric_checks:
        if field not in data_out:
            raise ValueError(f"Missing required field: '{field}'.")
        try:
            val = float(data_out[field])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid numeric input for '{field}': {data_out[field]}. Please enter a valid number.")

        if val < min_val or val > max_val:
            raise ValueError(f"{err_msg} Received: {val}.")

    # Categorical checks
    furnishing = str(data_out.get("furnishing", "")).strip()
    if not furnishing:
        data_out["furnishing"] = "Semi-Furnished"

    return data_out, location_intel


def explain_prediction(
    X_processed: np.ndarray,
    feature_names: List[str],
    weights: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Computes individual feature contributions: contribution_j = X_{i,j} * w_j.
    """
    contributions = X_processed[0] * weights
    indexed = list(enumerate(contributions))
    indexed.sort(key=lambda x: abs(x[1]), reverse=True)

    explanations = []
    for idx, contr in indexed[:top_k]:
        fname = feature_names[idx]
        is_positive = contr >= 0
        direction = "increased" if is_positive else "reduced"

        clean_name = fname.replace("_", " ").title()
        phrase = f"{clean_name} {direction} estimated value (contribution: {format_inr(contr)})."

        explanations.append(
            {
                "feature": fname,
                "display_name": clean_name,
                "contribution": contr,
                "is_positive": is_positive,
                "explanation": phrase,
            }
        )

    return explanations


def predict_house(
    raw_input_dict: Dict[str, Any],
    bundle: Dict[str, Any],
) -> Dict[str, Any]:
    """
    End-to-end inference function for a single house with automatic location feature enrichment.
    """
    # 1. Automatic location feature enrichment & validation
    enriched_data, location_intel = enrich_and_validate_input(raw_input_dict)

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_names = bundle["feature_names"]
    test_metrics = bundle.get("test_metrics", {})
    rmse = test_metrics.get("rmse", 500000.0)

    # 2. Convert to DataFrame
    df_input = pd.DataFrame([enriched_data])

    # 3. Preprocess
    X_processed = preprocessor.transform(df_input)

    # 4. Predict
    raw_price = float(model.predict(X_processed)[0])
    raw_price = max(100000.0, raw_price)

    # 5. Price per sqft
    area = float(enriched_data["area_sqft"])
    price_per_sqft = raw_price / area

    # 6. Confidence interval
    lower_bound, upper_bound = compute_confidence_band(raw_price, rmse)

    # 7. Model-based explanation
    explanations = explain_prediction(X_processed, feature_names, model.weights, top_k=5)

    return {
        "raw_price": raw_price,
        "formatted_price": format_inr(raw_price),
        "raw_price_per_sqft": price_per_sqft,
        "formatted_price_per_sqft": f"{format_inr(price_per_sqft)} / sq.ft",
        "lower_bound": lower_bound,
        "upper_bound": upper_bound,
        "formatted_lower_bound": format_inr(lower_bound),
        "formatted_upper_bound": format_inr(upper_bound),
        "rmse": rmse,
        "formatted_rmse": format_inr(rmse),
        "explanations": explanations,
        "location_intel": location_intel,
        "enriched_input": enriched_data,
    }
