"""
Feature Engineering Module for House Price Prediction.
Derives domain-relevant predictive features strictly avoiding target leakage.
Target variable ('price') is NEVER used in any feature derivation.
"""

from typing import Optional
import numpy as np
import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates derived features from raw property characteristics.

    Engineered Features:
    - total_rooms: Sum of bedrooms and bathrooms.
    - bed_to_bath_ratio: Ratio of bedrooms to bathrooms.
    - space_per_room: Floor area allocated per room.
    - amenity_score: Composite score summing garden, pool, and parking availability.
    - distance_score: Weighted accessibility index based on distances to city, school, and hospital.
    - is_new_property: Binary indicator for properties aged <= 3 years.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame containing raw features.

    Returns
    -------
    pd.DataFrame
        Copy of DataFrame augmented with engineered features.
    """
    df_out = df.copy()

    # Total rooms
    bedrooms = df_out["bedrooms"].fillna(df_out["bedrooms"].median() if "bedrooms" in df_out else 2)
    bathrooms = df_out["bathrooms"].fillna(df_out["bathrooms"].median() if "bathrooms" in df_out else 1)
    df_out["total_rooms"] = bedrooms + bathrooms

    # Bedroom to bathroom ratio
    df_out["bed_to_bath_ratio"] = bedrooms / np.maximum(bathrooms, 1)

    # Space per room (sqft)
    if "area_sqft" in df_out:
        df_out["space_per_room"] = df_out["area_sqft"] / np.maximum(df_out["total_rooms"], 1)

    # Amenity composite score
    garden_flag = (df_out["has_garden"].astype(str).str.lower().isin(["yes", "1", "true"])).astype(float)
    pool_flag = (df_out["has_pool"].astype(str).str.lower().isin(["yes", "1", "true"])).astype(float)
    parking_flag = (df_out["parking"].fillna(0) > 0).astype(float)
    df_out["amenity_score"] = garden_flag + pool_flag + parking_flag

    # Weighted distance score (lower is closer / more convenient)
    if (
        "distance_to_city_km" in df_out
        and "distance_to_school_km" in df_out
        and "distance_to_hospital_km" in df_out
    ):
        df_out["distance_score"] = (
            (df_out["distance_to_city_km"] * 0.50)
            + (df_out["distance_to_school_km"] * 0.25)
            + (df_out["distance_to_hospital_km"] * 0.25)
        )

    # Property age flag
    if "age_years" in df_out:
        df_out["is_new_property"] = (df_out["age_years"].fillna(10.0) <= 3.0).astype(float)

    return df_out
