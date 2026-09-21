"""
Unit tests for Preprocessing, Imputation, Scaling, and Outlier Handling.
"""

import numpy as np
import pandas as pd
import pytest

from src.data_loader import generate_synthetic_data
from src.preprocessing import DataPreprocessor


def test_missing_value_imputation():
    """Verify that numerical columns use median and categorical columns use mode."""
    df = pd.DataFrame(
        {
            "area_sqft": [1000.0, 2000.0, np.nan, 4000.0],
            "bedrooms": [2, 3, 4, np.nan],
            "bathrooms": [1, 2, 2, 3],
            "stories": [1, 2, 1, 2],
            "parking": [1, 1, 0, 2],
            "age_years": [5.0, np.nan, 15.0, 20.0],
            "distance_to_city_km": [5.0, 10.0, 15.0, 20.0],
            "distance_to_school_km": [2.0, 3.0, 1.0, 4.0],
            "distance_to_hospital_km": [3.0, 2.0, 4.0, 1.0],
            "crime_rate": [0.1, 0.2, np.nan, 0.4],
            "property_tax": [10000.0, 20000.0, 30000.0, 40000.0],
            "income_index": [0.5, 0.6, 0.7, 0.8],
            "location": ["Mumbai", "Bangalore", "Mumbai", np.nan],
            "furnishing": ["Furnished", "Unfurnished", "Furnished", "Furnished"],
            "has_garden": ["Yes", "No", "Yes", "No"],
            "has_pool": ["No", "No", "Yes", "No"],
        }
    )

    preprocessor = DataPreprocessor()
    X = preprocessor.fit_transform(df)

    assert not np.isnan(X).any(), "Processed feature matrix contains NaNs!"
    # Check that location mode was 'Mumbai'
    assert preprocessor.cat_modes["location"] == "Mumbai"
    # Check median of area_sqft [1000, 2000, 4000] is 2000
    assert preprocessor.num_medians["area_sqft"] == 2000.0


def test_outlier_capping():
    """Verify that IQR capping bounds values without dropping rows."""
    df = generate_synthetic_data(n_samples=200, random_state=42, introduce_outliers=True)
    initial_rows = len(df)

    preprocessor = DataPreprocessor(iqr_multiplier=1.5)
    X = preprocessor.fit_transform(df)

    # Must preserve row count
    assert X.shape[0] == initial_rows
    # Must have learned IQR bounds for all numerical features
    assert "area_sqft" in preprocessor.iqr_bounds
    lower, upper = preprocessor.iqr_bounds["area_sqft"]
    assert lower < upper


def test_no_data_leakage_in_scaling():
    """Ensure scaler fits ONLY on training data and doesn't update on test transform."""
    df_train = generate_synthetic_data(n_samples=500, random_state=42)
    df_test = generate_synthetic_data(n_samples=100, random_state=123)

    preprocessor = DataPreprocessor()
    preprocessor.fit(df_train)

    train_mean = preprocessor.standard_scaler.mean_.copy()

    # Transforming test data should not alter learned training means
    preprocessor.transform(df_test)
    test_transformed_mean = preprocessor.standard_scaler.mean_

    np.testing.assert_allclose(train_mean, test_transformed_mean)
