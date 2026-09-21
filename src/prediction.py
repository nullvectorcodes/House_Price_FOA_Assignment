"""
Inference and Real-World House Price Prediction Module.
Validates realistic constraints, applies preprocessor transforms,
and calculates price, price/sqft, confidence bands, and feature explanations.
"""

from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd

from src.evaluation import compute_confidence_band, format_inr


VALID_LOCATIONS = ["Mumbai", "Bangalore", "Delhi", "Hyderabad", "Pune", "Chennai"]
VALID_FURNISHING = ["Unfurnished", "Semi-Furnished", "Furnished"]


def validate_property_input(data: Dict[str, Any]) -> None:
    """
    Validates domain and physical constraints on house inputs.

    Raises
    ------
    ValueError
        If any input violates physical or domain boundaries.
    """
    # Numerical validation rules
    numeric_checks = [
        ("area_sqft", 100.0, 20000.0, "Area (sq.ft) must be a positive number greater than 100."),
        ("bedrooms", 1, 20, "Bedrooms must be an integer >= 1."),
        ("bathrooms", 1, 15, "Bathrooms must be an integer >= 1."),
        ("stories", 1, 10, "Stories must be an integer >= 1."),
        ("parking", 0, 10, "Parking spaces must be a non-negative integer (>= 0)."),
        ("age_years", 0.0, 150.0, "Property age must be a non-negative number (>= 0)."),
        ("distance_to_city_km", 0.0, 200.0, "Distance to city center must be >= 0 km."),
        ("distance_to_school_km", 0.0, 100.0, "Distance to school must be >= 0 km."),
        ("distance_to_hospital_km", 0.0, 100.0, "Distance to hospital must be >= 0 km."),
        ("crime_rate", 0.0, 1.0, "Crime rate must be between 0.0 and 1.0."),
        ("property_tax", 0.0, 1000000.0, "Property tax must be >= 0."),
        ("income_index", 0.0, 1.0, "Income index must be between 0.0 and 1.0."),
    ]

    for field, min_val, max_val, err_msg in numeric_checks:
        if field not in data:
            raise ValueError(f"Missing required field: '{field}'.")
        try:
            val = float(data[field])
        except (TypeError, ValueError):
            raise ValueError(f"Invalid numeric input for '{field}': {data[field]}. Please enter a valid number.")

        if val < min_val or val > max_val:
            raise ValueError(f"{err_msg} Received: {val}.")

    # Categorical checks
    location = str(data.get("location", "")).strip()
    if not location:
        raise ValueError("Location is required.")

    furnishing = str(data.get("furnishing", "")).strip()
    if not furnishing:
        raise ValueError("Furnishing status is required.")


def explain_prediction(
    X_processed: np.ndarray,
    feature_names: List[str],
    weights: np.ndarray,
    top_k: int = 5,
) -> List[Dict[str, Any]]:
    """
    Computes individual feature contributions: contribution_j = X_{i,j} * w_j.

    Note: These reflect model-based standardized contributions within the
    linear hypothesis space, and are not causal explanations.
    """
    contributions = X_processed[0] * weights
    indexed = list(enumerate(contributions))
    # Sort by absolute contribution magnitude
    indexed.sort(key=lambda x: abs(x[1]), reverse=True)

    explanations = []
    for idx, contr in indexed[:top_k]:
        fname = feature_names[idx]
        is_positive = contr >= 0
        direction = "increased" if is_positive else "reduced"

        # Human-readable phrasing
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
    End-to-end inference function for a single house.

    Parameters
    ----------
    raw_input_dict : Dict[str, Any]
        Raw input property features.
    bundle : Dict[str, Any]
        Model bundle containing 'model', 'preprocessor', 'feature_names', 'test_metrics'.

    Returns
    -------
    Dict[str, Any]
        Dictionary with formatted price, raw price, price per sqft,
        confidence interval bounds, and feature contributions.
    """
    # 1. Input Validation
    validate_property_input(raw_input_dict)

    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    feature_names = bundle["feature_names"]
    test_metrics = bundle.get("test_metrics", {})
    rmse = test_metrics.get("rmse", 500000.0)

    # 2. Convert to DataFrame
    df_input = pd.DataFrame([raw_input_dict])

    # 3. Preprocess
    X_processed = preprocessor.transform(df_input)

    # 4. Predict
    raw_price = float(model.predict(X_processed)[0])
    raw_price = max(100000.0, raw_price)  # Floor at 1 Lakh minimum

    # 5. Price per sqft
    area = float(raw_input_dict["area_sqft"])
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
    }
