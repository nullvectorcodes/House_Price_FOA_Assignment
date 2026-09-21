"""
Unit tests for Real-World Inference, Input Validation, and Persistence.
"""

import tempfile
from pathlib import Path
import joblib
import pytest

from src.data_loader import generate_synthetic_data
from src.linear_regression import RegularizedLinearRegression
from src.prediction import predict_house, validate_property_input
from src.preprocessing import DataPreprocessor


@pytest.fixture
def mock_bundle():
    """Generates a fitted model bundle for testing prediction."""
    df = generate_synthetic_data(n_samples=300, random_state=42)
    preprocessor = DataPreprocessor()
    X = preprocessor.fit_transform(df)
    y = df["price"].values

    model = RegularizedLinearRegression(learning_rate=0.01, n_epochs=100)
    model.fit(X, y)

    return {
        "model": model,
        "preprocessor": preprocessor,
        "feature_names": preprocessor.feature_names_,
        "test_metrics": {"rmse": 450000.0, "r2": 0.88, "mae": 320000.0},
    }


def test_valid_prediction(mock_bundle):
    """Verify that a valid property produces formatted prices and explanations."""
    sample_input = {
        "area_sqft": 1800.0,
        "bedrooms": 3,
        "bathrooms": 2,
        "stories": 2,
        "parking": 1,
        "age_years": 5.0,
        "distance_to_city_km": 8.0,
        "distance_to_school_km": 2.0,
        "distance_to_hospital_km": 3.0,
        "crime_rate": 0.20,
        "property_tax": 25000.0,
        "income_index": 0.75,
        "location": "Hyderabad",
        "furnishing": "Semi-Furnished",
        "has_garden": "Yes",
        "has_pool": "No",
    }

    res = predict_house(sample_input, mock_bundle)

    assert "raw_price" in res
    assert res["raw_price"] > 0
    assert "₹" in res["formatted_price"]
    assert "₹" in res["formatted_price_per_sqft"]
    assert len(res["explanations"]) == 5
    assert res["lower_bound"] < res["upper_bound"]


def test_invalid_input_validation():
    """Verify input validation catches boundary violations."""
    # Negative area
    with pytest.raises(ValueError, match="Area.*positive"):
        validate_property_input({"area_sqft": -500.0})

    # Zero bedrooms
    with pytest.raises(ValueError, match="Bedrooms"):
        validate_property_input({"area_sqft": 1500.0, "bedrooms": 0})

    # Negative distance
    with pytest.raises(ValueError, match="Distance"):
        validate_property_input(
            {
                "area_sqft": 1500.0,
                "bedrooms": 2,
                "bathrooms": 1,
                "stories": 1,
                "parking": 1,
                "age_years": 5,
                "distance_to_city_km": -5.0,
            }
        )


def test_model_serialization_reproducibility(mock_bundle):
    """Verify that serializing with joblib preserves identical predictions."""
    sample_input = {
        "area_sqft": 2200.0,
        "bedrooms": 4,
        "bathrooms": 3,
        "stories": 2,
        "parking": 2,
        "age_years": 3.0,
        "distance_to_city_km": 5.0,
        "distance_to_school_km": 1.5,
        "distance_to_hospital_km": 2.0,
        "crime_rate": 0.15,
        "property_tax": 35000.0,
        "income_index": 0.85,
        "location": "Bangalore",
        "furnishing": "Furnished",
        "has_garden": "Yes",
        "has_pool": "Yes",
    }

    res_before = predict_house(sample_input, mock_bundle)

    with tempfile.TemporaryDirectory() as tmpdir:
        bundle_path = Path(tmpdir) / "test_bundle.pkl"
        joblib.dump(mock_bundle, bundle_path)
        loaded_bundle = joblib.load(bundle_path)

        res_after = predict_house(sample_input, loaded_bundle)

    assert pytest.approx(res_before["raw_price"], rel=1e-6) == res_after["raw_price"]
