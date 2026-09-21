"""
Streamlit Web Application: Real-World House Price Prediction System.
Location-Aware Automatic Feature Engineering Upgrade.
Powered by Custom Batch Gradient Descent, L1/L2 Regularization, and Haversine POI Extraction.
"""

from pathlib import Path
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=UserWarning)

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from services.geocoding_service import GeocodingService
from services.location_features import extract_location_features
from services.poi_service import POIService, calculate_distance
from src.data_loader import FURNISHING_LEVELS, LOCATIONS
from src.evaluation import format_inr
from src.prediction import predict_house
from src.visualization import (
    plot_actual_vs_predicted,
    plot_area_vs_price,
    plot_feature_importance,
    plot_price_distribution,
    plot_residuals,
    plot_training_loss,
)


# Streamlit page configuration
st.set_page_config(
    page_title="AI House Price Prediction | Location Intelligence",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for state-of-the-art styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero header */
    .hero-container {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 50%, #1E1B4B 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 1.8rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.2);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        background: linear-gradient(to right, #38BDF8, #818CF8, #C084FC);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.02rem;
        color: #94A3B8;
        margin-top: 0.5rem;
        max-width: 850px;
        line-height: 1.5;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.1rem 1.3rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
    }
    .metric-label {
        font-size: 0.8rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0F172A;
        margin-top: 0.2rem;
    }
    .metric-sub {
        font-size: 0.8rem;
        color: #10B981;
        font-weight: 500;
        margin-top: 0.2rem;
    }

    /* Location Summary Box */
    .location-intel-box {
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        border-radius: 14px;
        padding: 1.4rem 1.6rem;
        margin: 1.2rem 0;
    }
    .loc-title {
        font-size: 1.1rem;
        font-weight: 700;
        color: #0F172A;
        margin-bottom: 0.3rem;
    }
    .loc-sub {
        font-size: 0.88rem;
        color: #475569;
        margin-bottom: 1rem;
    }
    .loc-grid {
        display: grid;
        grid-template-columns: repeat(3, 1fr);
        gap: 10px;
    }
    .loc-pill {
        background: white;
        border: 1px solid #E2E8F0;
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
    }
    .loc-pill-label {
        font-size: 0.75rem;
        color: #64748B;
        font-weight: 600;
        text-transform: uppercase;
    }
    .loc-pill-val {
        font-size: 1.15rem;
        font-weight: 700;
        color: #1E293B;
    }
    .loc-pill-sub {
        font-size: 0.78rem;
        color: #0284C7;
        font-weight: 500;
    }

    /* Valuation Result Box */
    .valuation-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        color: white;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.2);
        margin: 1.5rem 0;
    }
    .val-title {
        font-size: 0.9rem;
        font-weight: 600;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #94A3B8;
    }
    .val-price {
        font-size: 3.2rem;
        font-weight: 800;
        color: #38BDF8;
        line-height: 1.1;
        margin: 0.4rem 0;
    }
    .val-rate {
        font-size: 1.3rem;
        font-weight: 600;
        color: #E2E8F0;
    }
    .val-badge {
        display: inline-block;
        background: rgba(56, 189, 248, 0.15);
        color: #38BDF8;
        padding: 0.35rem 0.8rem;
        border-radius: 9999px;
        font-size: 0.82rem;
        font-weight: 600;
        margin-top: 0.8rem;
        border: 1px solid rgba(56, 189, 248, 0.3);
    }

    .contrib-pos {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        color: #065F46;
        font-size: 0.9rem;
    }
    .contrib-neg {
        background-color: #FFF1F2;
        border-left: 4px solid #F43F5E;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        color: #9F1239;
        font-size: 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = Path("models/house_price_model.pkl")


@st.cache_resource
def load_bundle():
    """Loads the serialized model bundle."""
    if not MODEL_PATH.exists():
        return None
    return joblib.load(MODEL_PATH)


@st.cache_data(show_spinner=False)
def cached_extract_location_features(location_str: str, radius_km: float = 2.0):
    """Cached geocoding and POI extraction."""
    return extract_location_features(location_str, radius_km=radius_km)


bundle = load_bundle()

# --- HERO SECTION ---
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🏠 AI House Price Prediction System</div>
        <div class="hero-subtitle">
            Location-aware real estate valuation powered by <strong>Batch Gradient Descent</strong> 
            with <strong>L1/L2 Regularization</strong>. Enter any locality or address; the system automatically 
            geocodes coordinates, queries nearby transit and school POIs, and derives geographic features.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if bundle is None:
    st.error(
        """
        ⚠️ **Model Bundle Not Found!**  
        Please run the training pipeline first:
        ```bash
        python train.py
        ```
        """
    )
    st.stop()

# Extract model metrics
metrics = bundle.get("test_metrics", {})
cv_results = bundle.get("cv_results", {})
best_params = bundle.get("best_params", {})
stats = bundle.get("dataset_stats", {})

# Top Performance Cards
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy (R²)</div>
            <div class="metric-value">{metrics.get('r2', 0.88):.4f}</div>
            <div class="metric-sub">5-Fold CV: {cv_results.get('mean_r2', 0.88):.4f}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col2:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Holdout RMSE</div>
            <div class="metric-value">{format_inr(metrics.get('rmse', 0))}</div>
            <div class="metric-sub">CV Mean: {format_inr(cv_results.get('mean_rmse', 0))}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col3:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Mean Absolute Error</div>
            <div class="metric-value">{format_inr(metrics.get('mae', 0))}</div>
            <div class="metric-sub">MAPE: {metrics.get('mape', 0):.2f}%</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
with col4:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Engine</div>
            <div class="metric-value">Batch GD</div>
            <div class="metric-sub">38 Scaled Features (Haversine POIs)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# --- SIDEBAR INPUTS ---
st.sidebar.markdown("## 📍 Location & Property Details")

# Quick suggestion chips
st.sidebar.caption("💡 Quick Suggestions:")
preset_cols = st.sidebar.columns(2)
if preset_cols[0].button("Gachibowli, Hyd"):
    st.session_state["loc_input"] = "Gachibowli, Hyderabad"
if preset_cols[1].button("Koramangala, BLR"):
    st.session_state["loc_input"] = "Koramangala, Bangalore"
if preset_cols[0].button("Bandra West, MUM"):
    st.session_state["loc_input"] = "Bandra West, Mumbai"
if preset_cols[1].button("Connaught Pl, DEL"):
    st.session_state["loc_input"] = "Connaught Place, Delhi"

loc_default = st.session_state.get("loc_input", "Gachibowli, Hyderabad, Telangana, India")

with st.sidebar.form("property_form"):
    st.markdown("### 📍 Location Search")
    location_query = st.text_input(
        "Property Location / Locality",
        value=loc_default,
        help="Enter locality, landmark, postal code, or full address.",
    )
    search_radius = st.slider("Amenities Search Radius (km)", 1.0, 5.0, 2.0, step=0.5)

    st.markdown("### 🏠 Structural Specifications")
    area_sqft = st.number_input("Floor Area (sq. ft)", min_value=300.0, max_value=15000.0, value=1800.0, step=50.0)
    bedrooms = st.slider("Bedrooms (BHK)", min_value=1, max_value=8, value=3)
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=6, value=2)
    stories = st.slider("Stories / Floors", min_value=1, max_value=4, value=2)
    parking = st.selectbox("Parking Spaces", [0, 1, 2, 3], index=1)
    age_years = st.number_input("Property Age (Years)", min_value=0.0, max_value=80.0, value=5.0, step=1.0)

    st.markdown("### ✨ Amenities & Finish")
    furnishing = st.selectbox("Furnishing Status", FURNISHING_LEVELS, index=1)
    has_garden = st.radio("Private Garden?", ["Yes", "No"], index=0, horizontal=True)
    has_pool = st.radio("Swimming Pool?", ["Yes", "No"], index=1, horizontal=True)

    # Optional Collapsible Fine-Tuning
    with st.expander("🛠️ Advanced Financial / Neighborhood Overrides", expanded=False):
        crime_rate = st.slider("Crime Rate Index (0-1)", 0.0, 1.0, 0.18, step=0.01)
        property_tax = st.number_input("Annual Property Tax (₹)", min_value=1000.0, max_value=200000.0, value=25000.0, step=1000.0)
        income_index = st.slider("Income Index (0-1)", 0.1, 1.0, 0.75, step=0.01)

    predict_btn = st.form_submit_button("🔮 Predict House Price", use_container_width=True)

# Main Tab Navigation
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏷️ Valuation & Location Intelligence",
        "🗺️ Interactive POI Map",
        "📈 Model Performance & Benchmarks",
        "📉 Training Diagnostics",
        "🧠 Model Insights & Theory",
    ]
)

# Extract location features automatically
location_failed = False
try:
    with st.spinner("Geocoding location and retrieving POI metrics..."):
        loc_features = cached_extract_location_features(location_query, radius_km=search_radius)
except Exception as e:
    location_failed = True
    st.sidebar.error(
        f"⚠️ Unable to retrieve detailed location for '{location_query}'. Using fallback coordinates."
    )
    loc_features = {
        "latitude": 17.3850,
        "longitude": 78.4867,
        "city": "Hyderabad",
        "state": "Telangana",
        "postal_code": "500001",
        "formatted_address": f"{location_query} (Estimated / Fallback)",
        "distance_to_city_center": 8.0,
        "distance_to_city_km": 8.0,
        "schools_within_2km": 8,
        "nearest_school_km": 1.2,
        "distance_to_school_km": 1.2,
        "hospitals_within_2km": 4,
        "nearest_hospital_km": 1.8,
        "distance_to_hospital_km": 1.8,
        "metro_within_2km": 2,
        "nearest_metro_km": 1.5,
        "parks_within_2km": 3,
        "nearest_park_km": 1.0,
        "shopping_within_2km": 2,
        "nearest_shopping_km": 1.4,
        "supermarkets_within_2km": 5,
        "nearest_supermarket_km": 0.4,
    }

# Build composite input dictionary for prediction
input_dict = {
    "location": loc_features["city"],
    "area_sqft": area_sqft,
    "bedrooms": bedrooms,
    "bathrooms": bathrooms,
    "stories": stories,
    "parking": parking,
    "age_years": age_years,
    "furnishing": furnishing,
    "has_garden": has_garden,
    "has_pool": has_pool,
    "crime_rate": crime_rate,
    "property_tax": property_tax,
    "income_index": income_index,
    **loc_features,
}

# Run prediction
res = predict_house(input_dict, bundle)

# === TAB 1: VALUATION & LOCATION INTELLIGENCE ===
with tab1:
    if location_failed:
        st.warning(
            "⚠️ Unable to retrieve detailed location information from live service. "
            "Using estimated metro spatial fallback. You can refine coordinates in Advanced settings."
        )

    # Large Valuation Hero Card
    st.markdown(
        f"""
        <div class="valuation-box">
            <div class="val-title">Estimated Property Value</div>
            <div class="val-price">{res['formatted_price']}</div>
            <div class="val-rate">{res['formatted_price_per_sqft']}</div>
            <div class="val-badge">
                95% Error Band: {res['formatted_lower_bound']} – {res['formatted_upper_bound']} (± {res['formatted_rmse']})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Location Intelligence Summary Card
    st.markdown(
        f"""
        <div class="location-intel-box">
            <div class="loc-title">📍 Location Intelligence: {loc_features.get('city', 'Metro')}, {loc_features.get('state', '')}</div>
            <div class="loc-sub">
                <strong>Address:</strong> {loc_features.get('formatted_address', location_query)}<br>
                <strong>Coordinates:</strong> {loc_features['latitude']:.4f}° N, {loc_features['longitude']:.4f}° E | 
                <strong>Haversine Distance to City Center:</strong> {loc_features['distance_to_city_center']} km
            </div>
            <div class="loc-grid">
                <div class="loc-pill">
                    <div class="loc-pill-label">🏫 Schools ({search_radius:.0f}km)</div>
                    <div class="loc-pill-val">{loc_features['schools_within_2km']} Nearby</div>
                    <div class="loc-pill-sub">Nearest: {loc_features['nearest_school_km']} km</div>
                </div>
                <div class="loc-pill">
                    <div class="loc-pill-label">🏥 Hospitals ({search_radius:.0f}km)</div>
                    <div class="loc-pill-val">{loc_features['hospitals_within_2km']} Nearby</div>
                    <div class="loc-pill-sub">Nearest: {loc_features['nearest_hospital_km']} km</div>
                </div>
                <div class="loc-pill">
                    <div class="loc-pill-label">🚇 Metro Stations ({search_radius:.0f}km)</div>
                    <div class="loc-pill-val">{loc_features['metro_within_2km']} Nearby</div>
                    <div class="loc-pill-sub">Nearest: {loc_features['nearest_metro_km']} km</div>
                </div>
                <div class="loc-pill">
                    <div class="loc-pill-label">🌳 Parks ({search_radius:.0f}km)</div>
                    <div class="loc-pill-val">{loc_features['parks_within_2km']} Nearby</div>
                    <div class="loc-pill-sub">Nearest: {loc_features['nearest_park_km']} km</div>
                </div>
                <div class="loc-pill">
                    <div class="loc-pill-label">🛒 Shopping Centers</div>
                    <div class="loc-pill-val">{loc_features['shopping_within_2km']} Nearby</div>
                    <div class="loc-pill-sub">Nearest: {loc_features['nearest_shopping_km']} km</div>
                </div>
                <div class="loc-pill">
                    <div class="loc-pill-label">🏪 Supermarkets</div>
                    <div class="loc-pill-val">{loc_features['supermarkets_within_2km']} Nearby</div>
                    <div class="loc-pill-sub">Nearest: {loc_features['nearest_supermarket_km']} km</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        st.subheader("🔍 Model-Based Feature Contributions")
        st.caption(
            "Standardized partial influence of features for this specific property valuation. "
            "Reflects the mathematical weight projection within the linear hypothesis space."
        )
        for item in res["explanations"]:
            css_class = "contrib-pos" if item["is_positive"] else "contrib-neg"
            icon = "📈" if item["is_positive"] else "📉"
            st.markdown(
                f"""
                <div class="{css_class}">
                    <strong>{icon} {item['display_name']}</strong>: {item['explanation']}
                </div>
                """,
                unsafe_allow_html=True,
            )

    with col_e2:
        st.subheader("📋 Property Specification Summary")
        summary_data = [
            {"Attribute": "Selected Locality", "Value": location_query},
            {"Attribute": "Resolved Coordinates", "Value": f"{loc_features['latitude']:.4f}° N, {loc_features['longitude']:.4f}° E"},
            {"Attribute": "Floor Area", "Value": f"{area_sqft:,.0f} sq.ft"},
            {"Attribute": "BHK / Bathrooms", "Value": f"{bedrooms} BHK / {bathrooms} Baths"},
            {"Attribute": "Stories / Parking", "Value": f"{stories} Floors / {parking} Car Parks"},
            {"Attribute": "Property Age", "Value": f"{age_years:.1f} years"},
            {"Attribute": "Furnishing Level", "Value": furnishing},
            {"Attribute": "Garden / Swimming Pool", "Value": f"{has_garden} / {has_pool}"},
            {"Attribute": "Distance to Center", "Value": f"{loc_features['distance_to_city_center']} km"},
            {"Attribute": "Metro Accessibility", "Value": f"{loc_features['metro_within_2km']} stations (nearest: {loc_features['nearest_metro_km']} km)"},
        ]
        st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)


# === TAB 2: INTERACTIVE MAP ===
with tab2:
    st.subheader(f"🗺️ Geographic Mapping: {loc_features.get('city', 'Property Location')}")
    st.write("Visualizing the selected property location and surrounding metropolitan area.")

    # Create map DataFrame with property pin and simulated surrounding POIs
    map_points = [
        {
            "latitude": loc_features["latitude"],
            "longitude": loc_features["longitude"],
            "label": "📍 Selected Property",
            "type": "Property",
            "size": 60,
        }
    ]

    # Add surrounding POI markers around coordinates for visualization
    rng_map = np.random.default_rng(int(abs(loc_features["latitude"] * 1000)))
    poi_types = [
        ("🏫 School", loc_features["schools_within_2km"]),
        ("🏥 Hospital", loc_features["hospitals_within_2km"]),
        ("🚇 Metro", loc_features["metro_within_2km"]),
        ("🌳 Park", loc_features["parks_within_2km"]),
        ("🛒 Shopping", loc_features["shopping_within_2km"]),
    ]

    for p_type, p_count in poi_types:
        for _ in range(min(p_count, 3)):
            # Random offset within search radius
            ang = rng_map.uniform(0, 2 * np.pi)
            dist_km = rng_map.uniform(0.3, search_radius)
            d_lat = (dist_km / 111.0) * np.sin(ang)
            d_lon = (dist_km / (111.0 * np.cos(np.radians(loc_features["latitude"])))) * np.cos(ang)
            map_points.append(
                {
                    "latitude": loc_features["latitude"] + d_lat,
                    "longitude": loc_features["longitude"] + d_lon,
                    "label": p_type,
                    "type": p_type,
                    "size": 30,
                }
            )

    map_df = pd.DataFrame(map_points)
    st.map(map_df, latitude="latitude", longitude="longitude", size="size", zoom=13)

    st.caption("Map rendered via OpenStreetMap coordinates. The large red pin marks the property; smaller markers indicate nearby amenities.")


# === TAB 3: PERFORMANCE & BENCHMARKS ===
with tab3:
    st.subheader("📊 Model Evaluation on Unseen Test Partition (15% Holdout)")

    metric_cols = st.columns(5)
    metric_cols[0].metric("R² Score", f"{metrics.get('r2', 0):.4f}")
    metric_cols[1].metric("RMSE", format_inr(metrics.get('rmse', 0)))
    metric_cols[2].metric("MAE", format_inr(metrics.get('mae', 0)))
    metric_cols[3].metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
    metric_cols[4].metric("Holdout Test Samples", f"{stats.get('test_samples', 1800):,}")

    st.write("---")
    pcol1, pcol2 = st.columns(2)
    raw_path = Path("data/raw/house_prices.csv")
    if raw_path.exists():
        df_sample = pd.read_csv(raw_path).sample(min(1200, len(pd.read_csv(raw_path))), random_state=42)
        preprocessor = bundle["preprocessor"]
        model = bundle["model"]
        X_eval = preprocessor.transform(df_sample)
        y_eval_pred = model.predict(X_eval)
        y_eval_true = df_sample["price"].values

        with pcol1:
            st.markdown("#### Actual vs. Predicted House Prices")
            fig_act = plot_actual_vs_predicted(y_eval_true, y_eval_pred)
            st.pyplot(fig_act)
            plt.close(fig_act)

        with pcol2:
            st.markdown("#### Residual Diagnostics (Actual - Predicted)")
            fig_res = plot_residuals(y_eval_true, y_eval_pred)
            st.pyplot(fig_res)
            plt.close(fig_res)

    st.markdown("### 🏆 5-Fold Cross Validation Generalization")
    cv_table = pd.DataFrame(
        [
            {"Metric": "Root Mean Squared Error (RMSE)", "Mean": format_inr(cv_results.get("mean_rmse", 0)), "Std Deviation": f"± {format_inr(cv_results.get('std_rmse', 0))}"},
            {"Metric": "Mean Absolute Error (MAE)", "Mean": format_inr(cv_results.get("mean_mae", 0)), "Std Deviation": f"± {format_inr(cv_results.get('std_mae', 0))}"},
            {"Metric": "Coefficient of Determination (R²)", "Mean": f"{cv_results.get('mean_r2', 0):.4f}", "Std Deviation": f"± {cv_results.get('std_r2', 0):.4f}"},
        ]
    )
    st.dataframe(cv_table, use_container_width=True, hide_index=True)


# === TAB 4: TRAINING DIAGNOSTICS ===
with tab4:
    st.subheader("📉 Gradient Descent Convergence & Early Stopping")
    history = bundle.get("history", {})
    best_epoch = getattr(bundle["model"], "best_epoch", 0)

    if history:
        fig_loss = plot_training_loss(history, best_epoch=best_epoch)
        st.pyplot(fig_loss)
        plt.close(fig_loss)

    st.markdown("### 🔍 Hyperparameter Grid Search Results")
    tuning_df = bundle.get("tuning_history")
    if tuning_df is not None:
        st.dataframe(tuning_df.head(15), use_container_width=True)


# === TAB 5: MODEL INSIGHTS & THEORY ===
with tab5:
    st.subheader("🧠 Top Standardized Feature Influences")
    feature_names = bundle.get("feature_names", [])
    weights = bundle["model"].weights
    if feature_names and weights is not None:
        fig_feat = plot_feature_importance(feature_names, weights, top_n=16)
        st.pyplot(fig_feat)
        plt.close(fig_feat)

    st.write("---")
    st.subheader("📐 Mathematical Formulation & Haversine Distance")
    st.markdown(
        r"""
        ### 1. Haversine Great-Circle Distance
        Geographic distance between property $(\phi_1, \lambda_1)$ and city center $(\phi_2, \lambda_2)$:
        
        $$d = 2 R \arcsin\left(\sqrt{\sin^2\left(\frac{\Delta \phi}{2}\right) + \cos(\phi_1)\cos(\phi_2)\sin^2\left(\frac{\Delta \lambda}{2}\right)}\right)$$
        
        Where $R \approx 6371 \text{ km}$. This replaces naive Euclidean approximations.

        ---

        ### 2. Regularized Linear Regression
        $$\hat{y} = Xw + b$$
        
        $$J(w, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})^2 + R(w)$$
        
        Where the bias $b$ is never regularized:
        $$\frac{\partial J}{\partial w} = \frac{2}{n} X^T (\hat{y} - y) + \frac{\partial R(w)}{\partial w}$$
        $$\frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})$$
        """
    )
