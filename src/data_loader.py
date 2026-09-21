"""
Data Loader and Synthetic Dataset Generator for House Price Prediction.
Generates 10,000+ realistic residential property records with realistic
economic relationships, correlations, Gaussian noise, and occasional missing values.
"""

from pathlib import Path
from typing import List, Optional
import numpy as np
import pandas as pd


EXPECTED_COLUMNS: List[str] = [
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "stories",
    "parking",
    "age_years",
    "distance_to_city_km",
    "distance_to_school_km",
    "distance_to_hospital_km",
    "crime_rate",
    "property_tax",
    "income_index",
    "location",
    "furnishing",
    "has_garden",
    "has_pool",
    "price",
]

LOCATIONS: List[str] = [
    "Mumbai",
    "Bangalore",
    "Delhi",
    "Hyderabad",
    "Pune",
    "Chennai",
]

FURNISHING_LEVELS: List[str] = ["Unfurnished", "Semi-Furnished", "Furnished"]


def generate_synthetic_data(
    n_samples: int = 12000,
    random_state: int = 42,
    introduce_missing: bool = True,
    introduce_outliers: bool = True,
) -> pd.DataFrame:
    """
    Generates a realistic housing dataset with Indian real estate pricing logic.

    Parameters
    ----------
    n_samples : int, default=12000 (exceeds the 10,000 minimum)
    random_state : int, default=42
    introduce_missing : bool, default=True
        Whether to introduce ~1-2% missing values in selected columns.
    introduce_outliers : bool, default=True
        Whether to add realistic mild outliers for IQR testing.

    Returns
    -------
    pd.DataFrame
        Housing dataset matching the required schema.
    """
    rng = np.random.default_rng(random_state)

    # 1. Base property attributes
    area_sqft = rng.normal(loc=1850, scale=650, size=n_samples)
    area_sqft = np.clip(area_sqft, 550, 5200)

    # Bedrooms correlated with area
    bedrooms = np.round(1 + (area_sqft / 800) + rng.normal(0, 0.4, n_samples))
    bedrooms = np.clip(bedrooms, 1, 6).astype(int)

    # Bathrooms correlated with bedrooms
    bathrooms = np.round(1 + 0.6 * bedrooms + rng.normal(0, 0.35, n_samples))
    bathrooms = np.clip(bathrooms, 1, 5).astype(int)

    stories = rng.choice([1, 2, 3, 4], size=n_samples, p=[0.45, 0.35, 0.15, 0.05])
    parking = rng.choice([0, 1, 2, 3], size=n_samples, p=[0.20, 0.50, 0.23, 0.07])
    age_years = np.clip(rng.exponential(scale=9.0, size=n_samples), 0, 45)

    # Distances
    distance_to_city_km = np.clip(rng.gamma(shape=2.5, scale=4.0, size=n_samples), 1.0, 38.0)
    distance_to_school_km = np.clip(rng.uniform(0.5, 12.0, size=n_samples), 0.5, 15.0)
    distance_to_hospital_km = np.clip(rng.uniform(0.5, 12.0, size=n_samples), 0.5, 15.0)

    # Crime rate (0 to 1 scale, skewed lower)
    crime_rate = np.clip(rng.beta(a=1.8, b=6.0, size=n_samples), 0.02, 0.95)

    # Income index (0.3 to 1.0)
    income_index = np.clip(rng.beta(a=4.0, b=2.5, size=n_samples), 0.25, 0.98)

    # Categorical features
    location = rng.choice(
        LOCATIONS,
        size=n_samples,
        p=[0.20, 0.22, 0.18, 0.16, 0.13, 0.11],
    )
    furnishing = rng.choice(
        FURNISHING_LEVELS,
        size=n_samples,
        p=[0.30, 0.45, 0.25],
    )

    # Amenities correlated with area and income index
    garden_prob = np.clip(0.15 + 0.25 * (area_sqft / 3000) + 0.2 * income_index, 0.1, 0.85)
    has_garden = np.where(rng.uniform(0, 1, size=n_samples) < garden_prob, "Yes", "No")

    pool_prob = np.clip(0.05 + 0.20 * (area_sqft / 3500) + 0.25 * income_index, 0.02, 0.65)
    has_pool = np.where(rng.uniform(0, 1, size=n_samples) < pool_prob, "Yes", "No")

    # City base price multiplier per sqft
    city_base_rate = {
        "Mumbai": 9500.0,
        "Bangalore": 6800.0,
        "Delhi": 7200.0,
        "Hyderabad": 5500.0,
        "Pune": 5200.0,
        "Chennai": 4900.0,
    }
    city_multiplier = np.array([city_base_rate[loc] for loc in location])

    # Furnishing value addition
    furnishing_add = np.array(
        [0.0 if f == "Unfurnished" else (250000.0 if f == "Semi-Furnished" else 600000.0) for f in furnishing]
    )

    # Garden & pool additions
    garden_add = np.where(has_garden == "Yes", 450000.0, 0.0)
    pool_add = np.where(has_pool == "Yes", 950000.0, 0.0)

    # Realistic price synthesis equation with non-linear factors and economic realism
    # Base structural value = area * city rate
    base_structure = area_sqft * city_multiplier

    # Room and level contributions
    room_val = (bedrooms * 220000.0) + (bathrooms * 160000.0) + (stories * 280000.0) + (parking * 180000.0)

    # Income multiplier effect (higher local income boosts property values by up to 25%)
    income_factor = 0.85 + (0.35 * income_index)

    # Depreciation due to age (up to ~25% max discount for old properties)
    age_depreciation = np.maximum(0.72, 1.0 - (0.009 * age_years))

    # Distance discounts (proximity to city center is highly valued)
    distance_discount = np.maximum(0.70, 1.0 - (0.012 * distance_to_city_km) - (0.004 * distance_to_school_km))

    # Crime rate penalty (high crime reduces value by up to 20%)
    crime_discount = 1.0 - (0.22 * crime_rate)

    # Combine economic value
    price = (
        (base_structure * income_factor * age_depreciation * distance_discount * crime_discount)
        + room_val
        + furnishing_add
        + garden_add
        + pool_add
    )

    # Property tax approximately proportional to price (~0.4% - 0.7% with noise)
    property_tax = np.clip(price * rng.uniform(0.004, 0.007, size=n_samples) + rng.normal(0, 1500, size=n_samples), 4500, 120000)

    # Realistic Gaussian noise (~5% standard deviation)
    noise = rng.normal(loc=0.0, scale=0.05 * price, size=n_samples)
    price = np.round(price + noise)
    price = np.clip(price, 1500000.0, 50000000.0)  # Bound between 15 Lakhs and 5 Crores

    df = pd.DataFrame(
        {
            "area_sqft": np.round(area_sqft, 1),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "stories": stories,
            "parking": parking,
            "age_years": np.round(age_years, 1),
            "distance_to_city_km": np.round(distance_to_city_km, 2),
            "distance_to_school_km": np.round(distance_to_school_km, 2),
            "distance_to_hospital_km": np.round(distance_to_hospital_km, 2),
            "crime_rate": np.round(crime_rate, 4),
            "property_tax": np.round(property_tax, 0),
            "income_index": np.round(income_index, 4),
            "location": location,
            "furnishing": furnishing,
            "has_garden": has_garden,
            "has_pool": has_pool,
            "price": price,
        }
    )

    # Introduce subtle realistic missing values (~1% in a few columns)
    if introduce_missing:
        # Age missing
        mask_age = rng.uniform(0, 1, size=n_samples) < 0.015
        df.loc[mask_age, "age_years"] = np.nan

        # Crime rate missing
        mask_crime = rng.uniform(0, 1, size=n_samples) < 0.012
        df.loc[mask_crime, "crime_rate"] = np.nan

        # Furnishing missing
        mask_furn = rng.uniform(0, 1, size=n_samples) < 0.010
        df.loc[mask_furn, "furnishing"] = np.nan

    # Introduce realistic outliers for IQR testing
    if introduce_outliers:
        outlier_indices = rng.choice(n_samples, size=30, replace=False)
        # Extremely large penthouse or luxury estate
        df.loc[outlier_indices[:15], "area_sqft"] = rng.uniform(5500, 7500, size=15)
        df.loc[outlier_indices[:15], "price"] = rng.uniform(42000000, 68000000, size=15)

    return df


def load_data(filepath: str | Path) -> pd.DataFrame:
    """
    Loads dataset from CSV and verifies schema integrity.

    Parameters
    ----------
    filepath : str or Path

    Returns
    -------
    pd.DataFrame
    """
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found at: {path}")

    df = pd.read_csv(path)
    missing_cols = [col for col in EXPECTED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Dataset at {path} is missing expected columns: {missing_cols}"
        )

    return df
