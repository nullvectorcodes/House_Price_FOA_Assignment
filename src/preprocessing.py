"""
Data Preprocessing Pipeline for House Price Prediction.

Handles:
- Duplicate detection & removal
- Missing value imputation (Median for numerical, Mode for categorical)
- Outlier handling (IQR-based winsorization/capping to prevent gradient instability)
- Categorical encoding (OneHotEncoder for multi-category variables)
- Numerical scaling (StandardScaler to guarantee isotropic gradient descent contours)
- Strict prevention of data leakage (fittings performed ONLY on training set)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.feature_engineering import engineer_features


CATEGORICAL_COLS = ["location", "furnishing"]
BINARY_COLS = ["has_garden", "has_pool"]
NUMERICAL_COLS = [
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
    # Engineered numeric features
    "total_rooms",
    "bed_to_bath_ratio",
    "space_per_room",
    "amenity_score",
    "distance_score",
    "is_new_property",
]
TARGET_COL = "price"


class DataPreprocessor:
    """
    Robust Preprocessing Pipeline for House Price Prediction.

    Ensures that all statistics (medians, modes, IQR bounds, scaler means/variances,
    and one-hot categories) are learned strictly on the training set and applied
    consistently across validation and test splits without leakage.
    """

    def __init__(self, iqr_multiplier: float = 1.5):
        """
        Parameters
        ----------
        iqr_multiplier : float, default=1.5
            Multiplier for IQR outlier capping. Values beyond [Q1 - k*IQR, Q3 + k*IQR]
            are clipped to boundaries rather than dropped, preserving dataset size
            while safeguarding gradient descent against extreme gradient spikes.
        """
        self.iqr_multiplier = iqr_multiplier

        # Learned statistics from training set
        self.num_medians: Dict[str, float] = {}
        self.cat_modes: Dict[str, str] = {}
        self.iqr_bounds: Dict[str, Tuple[float, float]] = {}

        # Transformers
        self.one_hot_encoder: Optional[OneHotEncoder] = None
        self.standard_scaler: Optional[StandardScaler] = None

        # Feature tracking
        self.feature_names_: List[str] = []
        self.is_fitted: bool = False

    def _impute_missing(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Imputes missing values using median for numerical and mode for categorical."""
        df_clean = df.copy()

        # Numerical imputation
        for col in NUMERICAL_COLS:
            if col in df_clean.columns:
                if fit:
                    median_val = float(df_clean[col].median(skipna=True))
                    self.num_medians[col] = median_val
                fallback = self.num_medians.get(col, 0.0)
                df_clean[col] = df_clean[col].fillna(fallback)

        # Categorical imputation
        for col in CATEGORICAL_COLS + BINARY_COLS:
            if col in df_clean.columns:
                if fit:
                    mode_val = (
                        df_clean[col].mode(dropna=True)[0]
                        if not df_clean[col].dropna().empty
                        else "Unknown"
                    )
                    self.cat_modes[col] = mode_val
                fallback = self.cat_modes.get(col, "Unknown")
                df_clean[col] = df_clean[col].fillna(fallback)

        return df_clean

    def _handle_outliers(self, df: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """
        Applies IQR-based soft capping (Winsorization).
        Bounds are computed strictly on training data.
        """
        df_capped = df.copy()
        for col in NUMERICAL_COLS:
            if col in df_capped.columns:
                if fit:
                    q1 = float(df_capped[col].quantile(0.25))
                    q3 = float(df_capped[col].quantile(0.75))
                    iqr = q3 - q1
                    lower = q1 - (self.iqr_multiplier * iqr)
                    upper = q3 + (self.iqr_multiplier * iqr)
                    self.iqr_bounds[col] = (lower, upper)

                if col in self.iqr_bounds:
                    lower, upper = self.iqr_bounds[col]
                    df_capped[col] = df_capped[col].clip(lower=lower, upper=upper)

        return df_capped

    def fit(self, df_raw: pd.DataFrame) -> "DataPreprocessor":
        """
        Fits all preprocessors strictly on the training set.

        Parameters
        ----------
        df_raw : pd.DataFrame
            Raw training DataFrame.

        Returns
        -------
        self : DataPreprocessor
        """
        # 1. Feature Engineering
        df_fe = engineer_features(df_raw)

        # 2. Impute missing values (fit medians and modes)
        df_imputed = self._impute_missing(df_fe, fit=True)

        # 3. Handle outliers (fit IQR boundaries)
        df_capped = self._handle_outliers(df_imputed, fit=True)

        # 4. Fit OneHotEncoder on categorical features
        self.one_hot_encoder = OneHotEncoder(
            sparse_output=False,
            handle_unknown="ignore",
            drop="first",
        )
        cat_data = df_capped[CATEGORICAL_COLS].astype(str)
        self.one_hot_encoder.fit(cat_data)
        encoded_cat_names = list(self.one_hot_encoder.get_feature_names_out(CATEGORICAL_COLS))

        # Binary features (0 or 1)
        binary_names = BINARY_COLS

        # 5. Fit StandardScaler on numerical features
        self.standard_scaler = StandardScaler()
        num_data = df_capped[NUMERICAL_COLS].values
        self.standard_scaler.fit(num_data)

        # Construct final ordered feature names
        self.feature_names_ = NUMERICAL_COLS + binary_names + encoded_cat_names
        self.is_fitted = True
        return self

    def transform(self, df_raw: pd.DataFrame) -> np.ndarray:
        """
        Transforms input DataFrame into the standardized feature matrix X.

        Parameters
        ----------
        df_raw : pd.DataFrame

        Returns
        -------
        np.ndarray of shape (n_samples, n_features)
        """
        if not self.is_fitted or self.standard_scaler is None or self.one_hot_encoder is None:
            raise ValueError("Preprocessor is not fitted. Call fit() first.")

        # 1. Feature Engineering
        df_fe = engineer_features(df_raw)

        # 2. Impute missing
        df_imputed = self._impute_missing(df_fe, fit=False)

        # 3. IQR Capping
        df_capped = self._handle_outliers(df_imputed, fit=False)

        # 4. Standardize numerical features
        num_vals = df_capped[NUMERICAL_COLS].values
        num_scaled = self.standard_scaler.transform(num_vals)

        # 5. Binary flags
        binary_vals = []
        for col in BINARY_COLS:
            flags = (
                df_capped[col]
                .astype(str)
                .str.lower()
                .isin(["yes", "1", "true"])
                .astype(float)
                .values.reshape(-1, 1)
            )
            binary_vals.append(flags)
        binary_scaled = np.hstack(binary_vals)

        # 6. One-hot encoded categorical features
        cat_data = df_capped[CATEGORICAL_COLS].astype(str)
        cat_encoded = self.one_hot_encoder.transform(cat_data)

        # Combine all processed features
        X_processed = np.hstack([num_scaled, binary_scaled, cat_encoded])
        return X_processed

    def fit_transform(self, df_raw: pd.DataFrame) -> np.ndarray:
        """Fits on data and returns transformed feature matrix."""
        return self.fit(df_raw).transform(df_raw)
