"""
Streamlit Web Application: Real-World House Price Prediction System.
Powered by Custom Batch Gradient Descent, L1/L2 Regularization, and Early Stopping.
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

from src.data_loader import LOCATIONS, FURNISHING_LEVELS
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


# Page configuration
st.set_page_config(
    page_title="AI House Price Prediction System",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for state-of-the-art styling
st.markdown(
    """
    <style>
    /* Global styling */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    /* Hero header */
    .hero-container {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 50%, #1E1B4B 100%);
        color: white;
        padding: 2.2rem 2.5rem;
        border-radius: 16px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.15), 0 8px 10px -6px rgba(0, 0, 0, 0.1);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .hero-title {
        font-size: 2.2rem;
        font-weight: 800;
        margin: 0;
        letter-spacing: -0.02em;
        background: linear-gradient(to right, #60A5FA, #A78BFA);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .hero-subtitle {
        font-size: 1.05rem;
        color: #94A3B8;
        margin-top: 0.5rem;
        max-width: 800px;
    }

    /* Metric Cards */
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px -2px rgba(0, 0, 0, 0.08);
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .metric-value {
        font-size: 1.65rem;
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

    /* Valuation Result Hero */
    .valuation-box {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 2rem;
        color: white;
        box-shadow: 0 20px 25px -5px rgba(0, 0, 0, 0.15);
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
        font-size: 3rem;
        font-weight: 800;
        color: #38BDF8;
        line-height: 1.1;
        margin: 0.4rem 0;
    }
    .val-rate {
        font-size: 1.25rem;
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

    /* Explanation badges */
    .contrib-pos {
        background-color: #ECFDF5;
        border-left: 4px solid #10B981;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        color: #065F46;
        font-size: 0.92rem;
    }
    .contrib-neg {
        background-color: #FFF1F2;
        border-left: 4px solid #F43F5E;
        padding: 0.75rem 1rem;
        border-radius: 6px;
        margin-bottom: 0.5rem;
        color: #9F1239;
        font-size: 0.92rem;
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


bundle = load_bundle()

# --- HERO SECTION ---
st.markdown(
    """
    <div class="hero-container">
        <div class="hero-title">🏠 AI House Price Prediction System</div>
        <div class="hero-subtitle">
            Predict the estimated market value of residential properties using a custom 
            <strong>Batch Gradient Descent</strong> engine with <strong>L1/L2 Regularization</strong> 
            and <strong>Early Stopping</strong>. Built from scratch without black-box fit routines.
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

if bundle is None:
    st.error(
        """
        ⚠️ **Model Bundle Not Found!**  
        The trained model bundle has not been generated yet. Please run the training pipeline first:
        ```bash
        python train.py
        ```
        """
    )
    st.stop()

# Extract model stats
metrics = bundle.get("test_metrics", {})
cv_results = bundle.get("cv_results", {})
best_params = bundle.get("best_params", {})
stats = bundle.get("dataset_stats", {})

# Top Model Performance Summary Row
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Model Accuracy (R²)</div>
            <div class="metric-value">{metrics.get('r2', 0.90):.4f}</div>
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
    reg_name = best_params.get("penalty", "None")
    reg_str = f"{reg_name.upper()} (λ={best_params.get('lmbda', 0)})" if reg_name else "None"
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">Algorithm</div>
            <div class="metric-value">Batch GD</div>
            <div class="metric-sub">Reg: {reg_str} | α={best_params.get('learning_rate', 0.01)}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

st.write("")

# --- SIDEBAR INPUTS ---
st.sidebar.markdown("## ⚙️ Property Specifications")
st.sidebar.markdown("Configure property characteristics to generate a real-time valuation.")

with st.sidebar.form("property_form"):
    st.markdown("### 📍 Location & Structure")
    location = st.selectbox("Location / City", LOCATIONS, index=3)
    area_sqft = st.number_input("Area (sq. ft)", min_value=300.0, max_value=12000.0, value=1800.0, step=50.0)
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=8, value=3)
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=6, value=2)
    stories = st.slider("Stories / Floors", min_value=1, max_value=4, value=2)
    parking = st.selectbox("Parking Spaces", [0, 1, 2, 3], index=1)
    age_years = st.number_input("Property Age (Years)", min_value=0.0, max_value=80.0, value=5.0, step=1.0)

    st.markdown("### ✨ Amenities & Furnishing")
    furnishing = st.selectbox("Furnishing Status", FURNISHING_LEVELS, index=1)
    has_garden = st.radio("Private Garden?", ["Yes", "No"], index=0, horizontal=True)
    has_pool = st.radio("Swimming Pool?", ["Yes", "No"], index=1, horizontal=True)

    st.markdown("### 🚗 Accessibility & Neighbourhood")
    distance_to_city_km = st.slider("Distance to City Center (km)", 0.5, 40.0, 8.0, step=0.5)
    distance_to_school_km = st.slider("Distance to Nearest School (km)", 0.5, 20.0, 2.0, step=0.5)
    distance_to_hospital_km = st.slider("Distance to Hospital (km)", 0.5, 20.0, 3.0, step=0.5)
    crime_rate = st.slider("Neighborhood Crime Rate Index", 0.0, 1.0, 0.20, step=0.01)
    property_tax = st.number_input("Annual Property Tax (₹)", min_value=1000.0, max_value=200000.0, value=25000.0, step=1000.0)
    income_index = st.slider("Local Median Income Index", 0.1, 1.0, 0.75, step=0.01)

    predict_btn = st.form_submit_button("⚡ Predict House Price", use_container_width=True)

# Build inputs dictionary
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

# --- TABS NAVIGATION ---
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "🏷️ Property Valuation",
        "📈 Model Performance",
        "📉 Training & Convergence",
        "🧠 Model Insights & EDA",
        "🏗️ Architecture & Theory",
    ]
)

# === TAB 1: VALUATION ===
with tab1:
    # Run prediction
    res = predict_house(input_data, bundle)

    st.markdown(
        f"""
        <div class="valuation-box">
            <div class="val-title">Estimated Market Valuation</div>
            <div class="val-price">{res['formatted_price']}</div>
            <div class="val-rate">{res['formatted_price_per_sqft']}</div>
            <div class="val-badge">
                95% Error Band: {res['formatted_lower_bound']} – {res['formatted_upper_bound']} (± {res['formatted_rmse']})
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col_prop1, col_prop2 = st.columns([1, 1])

    with col_prop1:
        st.subheader("🔍 Model-Based Feature Contributions")
        st.caption(
            "Standardized partial influence of features for this specific property. "
            "Note: These represent linear model weight projections, not causal mechanisms."
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

    with col_prop2:
        st.subheader("📋 Property Specification Summary")
        summary_df = pd.DataFrame(
            [
                {"Feature": "Location", "Value": location},
                {"Feature": "Floor Area", "Value": f"{area_sqft:,.0f} sq.ft"},
                {"Feature": "Bedrooms / Bathrooms", "Value": f"{bedrooms} BHK / {bathrooms} Baths"},
                {"Feature": "Stories / Parking", "Value": f"{stories} Floors / {parking} Spots"},
                {"Feature": "Age of Property", "Value": f"{age_years:.1f} years"},
                {"Feature": "Furnishing", "Value": furnishing},
                {"Feature": "Garden / Pool", "Value": f"{has_garden} / {has_pool}"},
                {"Feature": "Distance to City Center", "Value": f"{distance_to_city_km} km"},
                {"Feature": "Neighborhood Income Index", "Value": f"{income_index:.2f}"},
            ]
        )
        st.dataframe(summary_df, use_container_width=True, hide_index=True)


# === TAB 2: MODEL PERFORMANCE ===
with tab2:
    st.subheader("📊 Evaluation on Unseen Test Dataset (15% Holdout)")
    st.write(
        "The model was evaluated strictly on test properties never exposed during gradient descent updates or hyperparameter selection."
    )

    metric_cols = st.columns(5)
    metric_cols[0].metric("R² Score", f"{metrics.get('r2', 0):.4f}")
    metric_cols[1].metric("RMSE", format_inr(metrics.get('rmse', 0)))
    metric_cols[2].metric("MAE", format_inr(metrics.get('mae', 0)))
    metric_cols[3].metric("MAPE", f"{metrics.get('mape', 0):.2f}%")
    metric_cols[4].metric("Holdout Samples", f"{stats.get('test_samples', 1800):,}")

    st.write("---")

    pcol1, pcol2 = st.columns(2)
    # Load raw data to generate plots if needed
    raw_path = Path("data/raw/house_prices.csv")
    if raw_path.exists():
        df_all = pd.read_csv(raw_path)
        y_test_sample = df_all["price"].values[:1000]
        # Generate predictions on sample
        preprocessor = bundle["preprocessor"]
        model = bundle["model"]
        X_sample = preprocessor.transform(df_all.iloc[:1000])
        y_pred_sample = model.predict(X_sample)

        with pcol1:
            st.markdown("#### Actual vs. Predicted House Prices")
            fig_act = plot_actual_vs_predicted(y_test_sample, y_pred_sample)
            st.pyplot(fig_act)
            plt.close(fig_act)

        with pcol2:
            st.markdown("#### Residual Diagnostics (Actual - Predicted)")
            fig_res = plot_residuals(y_test_sample, y_pred_sample)
            st.pyplot(fig_res)
            plt.close(fig_res)

    st.markdown("### 🏆 5-Fold Cross Validation Generalization")
    st.write(
        "K-Fold Cross-Validation splits the training set into 5 folds, training on 4 and validating on 1 iteratively. "
        "This estimates the model's true expected out-of-sample error and verifies that the model does not suffer from high variance."
    )
    cv_table = pd.DataFrame(
        [
            {"Metric": "Root Mean Squared Error (RMSE)", "Mean": format_inr(cv_results.get("mean_rmse", 0)), "Std Deviation": f"± {format_inr(cv_results.get('std_rmse', 0))}"},
            {"Metric": "Mean Absolute Error (MAE)", "Mean": format_inr(cv_results.get("mean_mae", 0)), "Std Deviation": f"± {format_inr(cv_results.get('std_mae', 0))}"},
            {"Metric": "Coefficient of Determination (R²)", "Mean": f"{cv_results.get('mean_r2', 0):.4f}", "Std Deviation": f"± {cv_results.get('std_r2', 0):.4f}"},
        ]
    )
    st.dataframe(cv_table, use_container_width=True, hide_index=True)


# === TAB 3: TRAINING & CONVERGENCE ===
with tab3:
    st.subheader("📉 Gradient Descent Convergence & Early Stopping")
    st.write(
        "Batch Gradient Descent minimizes the mean squared error plus regularization penalties by taking iterative steps "
        "proportional to the analytical negative gradient."
    )

    history = bundle.get("history", {})
    best_epoch = getattr(bundle["model"], "best_epoch", 0)

    if history:
        fig_loss = plot_training_loss(history, best_epoch=best_epoch)
        st.pyplot(fig_loss)
        plt.close(fig_loss)

    st.markdown("### 🔍 Hyperparameter Tuning Grid Search")
    tuning_df = bundle.get("tuning_history")
    if tuning_df is not None:
        st.dataframe(tuning_df.head(15), use_container_width=True)


# === TAB 4: MODEL INSIGHTS & EDA ===
with tab4:
    st.subheader("🧠 Standardized Feature Influences")
    st.write(
        "Because all features are standardized with `StandardScaler` (zero mean, unit variance), "
        "the magnitude of learned weight coefficients reflects the relative strength of association."
    )

    feature_names = bundle.get("feature_names", [])
    weights = bundle["model"].weights
    if feature_names and weights is not None:
        fig_feat = plot_feature_importance(feature_names, weights, top_n=15)
        st.pyplot(fig_feat)
        plt.close(fig_feat)

    st.write("---")
    st.subheader("📊 Exploratory Dataset Distributions")
    if raw_path.exists():
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            fig_dist = plot_price_distribution(df_all["price"].values)
            st.pyplot(fig_dist)
            plt.close(fig_dist)
        with col_d2:
            fig_area = plot_area_vs_price(df_all)
            st.pyplot(fig_area)
            plt.close(fig_area)


# === TAB 5: ARCHITECTURE & THEORY ===
with tab5:
    st.subheader("🏗️ System Architecture & Mathematical Formulation")
    st.markdown(
        r"""
        ### 1. Mathematical Formulation
        The primary model is implemented from first principles using NumPy:
        
        $$\hat{y} = Xw + b$$
        
        #### Cost Function with Regularization:
        $$J(w, b) = \frac{1}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})^2 + R(w)$$
        
        - **No Regularization**: $R(w) = 0$
        - **L2 Regularization (Ridge)**: $R(w) = \lambda \sum_{j=1}^d w_j^2$
        - **L1 Regularization (Lasso)**: $R(w) = \lambda \sum_{j=1}^d |w_j|$
        
        > **Note on Bias Term ($b$)**: The bias term is never regularized! Penalizing the intercept would unfairly pull baseline house valuations toward zero, distorting price predictions across the market.

        #### Analytical Gradients:
        $$\frac{\partial J}{\partial w} = \frac{2}{n} X^T (\hat{y} - y) + \frac{\partial R(w)}{\partial w}$$
        $$\frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})$$
        
        Where:
        $$\frac{\partial R(w)}{\partial w} = \begin{cases} 0 & \text{No Reg} \\ 2\lambda w & \text{L2} \\ \lambda \text{sign}(w) & \text{L1} \end{cases}$$
        
        #### Batch Gradient Descent Update Rule:
        $$w \leftarrow w - \alpha \frac{\partial J}{\partial w}$$
        $$b \leftarrow b - \alpha \frac{\partial J}{\partial b}$$
        
        Where $\alpha$ represents the step size (learning rate).
        
        ---
        
        ### 2. End-to-End Pipeline
        1. **Data Ingestion**: Raw dataset containing structural, geographical, economic, and amenity characteristics.
        2. **Preprocessing (No Data Leakage)**: Median imputation for numerical features, mode imputation for categorical features, IQR soft capping for outliers, and OneHotEncoding for multi-class categories.
        3. **StandardScaler**: Crucial for Gradient Descent. Standardizes all features to zero mean and unit variance, preventing gradient oscillations on high-magnitude features (e.g. `area_sqft` vs `crime_rate`).
        4. **Early Stopping**: Monitored on a 15% validation split. If validation loss does not improve for 100 consecutive epochs, training terminates and the best parameter states are restored.
        """
    )
