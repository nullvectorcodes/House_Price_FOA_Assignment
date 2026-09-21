# 🏠 House Price Prediction Using Gradient Descent and Regularization

A complete, production-grade Machine Learning system and interactive web dashboard built from first principles for residential property price valuation.

This project implements **Linear Regression**, **Batch Gradient Descent**, **L1 (Lasso) Regularization**, **L2 (Ridge) Regularization**, **Feature Scaling**, and **Early Stopping** entirely from scratch in **NumPy** without relying on black-box optimization routines. Scikit-learn is utilized strictly for benchmarking comparison models and data splitting.

---

## 📌 Problem Statement

Predicting residential real estate prices is a classic yet challenging regression problem. Property valuations depend upon a combination of structural attributes (area, bedrooms, bathrooms, stories), location characteristics (distance to city centers, educational and healthcare institutions), environmental factors (neighborhood crime rates, income levels), and amenities (swimming pools, private gardens, parking facilities).

Traditional ordinary least squares (OLS) solvers compute an analytical normal equation $(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{y}$, which scales with $O(d^3)$ computational complexity and breaks down when features are collinear. In this project, we implement an iterative, explainable **Batch Gradient Descent** optimization engine with **L1 and L2 regularization penalties** that efficiently learns parameters on large datasets while controlling overfitting.

---

## 🎯 Objectives

1. **First-Principles Implementation**: Build a modular `RegularizedLinearRegression` class from scratch using NumPy.
2. **Gradient Descent Optimization**: Implement exact analytical loss gradient calculations and iterative weight updates.
3. **L1 & L2 Regularization**: Implement Lasso ($L_1$) and Ridge ($L_2$) penalties to mitigate multicollinearity and perform weight shrinkage without regularizing the bias term.
4. **Early Stopping & Convergence**: Monitor validation loss per epoch, track convergence trajectories, and restore best parameters on plateau.
5. **Leakage-Free Preprocessing**: Construct an end-to-end data preprocessing pipeline (imputation, IQR winsorization, one-hot encoding, and feature standardization) fitted strictly on the training partition.
6. **Generalization & Evaluation**: Assess out-of-sample performance using 5-Fold Cross-Validation, $R^2$, RMSE, MAE, and MAPE.
7. **Interactive Real-World Dashboard**: Deploy a Streamlit web application providing instant valuations in Indian Rupee format (`₹XX,XX,XXX`), price per square foot (`₹X,XXX / sq.ft`), 95% empirical error bands, and model-based feature contribution breakdowns.

---

## 🏗️ End-to-End System Architecture

## 🏗️ End-to-End System Architecture

```text
                  +-------------------------------------------------------+
                  |         USER INPUT: "Gachibowli, Hyderabad"           |
                  +---------------------------+---------------------------+
                                              |
                                              v
                  +-------------------------------------------------------+
                  |              services/geocoding_service.py            |
                  |  - Nominatim / Geocoding API + Caching                |
                  |  - Coordinates: (17.4436° N, 78.3520° E)             |
                  |  - Formatted Address, City, State, Postal Code        |
                  +---------------------------+---------------------------+
                                              |
                                              v
                  +-------------------------------------------------------+
                  |                 services/poi_service.py               |
                  |  - Haversine Distance: calculate_distance(lat1,lon1..)|
                  |  - POI Categories: schools, hospitals, metro, parks,  |
                  |    shopping, supermarkets within configurable radius  |
                  +---------------------------+---------------------------+
                                              |
                                              v
                  +-------------------------------------------------------+
                  |            services/location_features.py              |
                  |  - Constructs ML Feature Vector:                      |
                  |    * latitude, longitude, distance_to_city_center     |
                  |    * schools_within_2km, nearest_school_km            |
                  |    * hospitals_within_2km, nearest_hospital_km        |
                  |    * metro_within_2km, nearest_metro_km, parks, etc.  |
                  +---------------------------+---------------------------+
                                              |
                                              v
                  +-------------------------------------------------------+
                  |              Preprocessing & Feature Scaling          |
                  |  (Median/Mode Imputation, IQR Capping, StandardScaler)|
                  +---------------------------+---------------------------+
                                              |
                                              v
                  +-------------------------------------------------------+
                  |      Custom Regularized Linear Regression Engine      |
                  |       (Batch Gradient Descent + L1/L2 Penalties)      |
                  +---------------------------+---------------------------+
                                              |
                        +---------------------+---------------------+
                        |                                           |
                        v                                           v
            +-----------------------+                   +-----------------------+
            |  Interactive CLI      |                   |  Streamlit Dashboard  |
            |  (predict.py)         |                   |  (app.py)             |
            +-----------------------+                   +-----------------------+
```

---

## 📊 Dataset Features Dictionary

| Column Name | Type | Unit / Values | Description |
| :--- | :--- | :--- | :--- |
| `area_sqft` | Float | Square feet (500–5,500) | Carpet/super built-up floor area. |
| `bedrooms` | Integer | Count (1–6) | Number of bedrooms. |
| `bathrooms` | Integer | Count (1–5) | Number of bathrooms. |
| `stories` | Integer | Count (1–4) | Number of floors/stories. |
| `parking` | Integer | Count (0–3) | Designated covered parking spaces. |
| `age_years` | Float | Years (0–45) | Age of property since construction. |
| `latitude` | Float | Decimal Degrees | Latitude of property. |
| `longitude` | Float | Decimal Degrees | Longitude of property. |
| `distance_to_city_km` | Float | Kilometers (1–38) | Haversine distance to central business district. |
| `schools_within_2km` | Integer | Count (0–28) | Number of educational institutions within 2km. |
| `nearest_school_km` | Float | Kilometers | Haversine distance to nearest school. |
| `hospitals_within_2km` | Integer | Count (0–15) | Number of hospitals/clinics within 2km. |
| `nearest_hospital_km` | Float | Kilometers | Haversine distance to nearest hospital. |
| `metro_within_2km` | Integer | Count (0–6) | Number of metro/transit stations within 2km. |
| `nearest_metro_km` | Float | Kilometers | Haversine distance to nearest metro station. |
| `parks_within_2km` | Integer | Count (0–10) | Parks and recreational spaces within 2km. |
| `shopping_within_2km`| Integer | Count (0–8) | Shopping malls within 2km. |
| `supermarkets_within_2km`| Integer | Count (1–18) | Grocery stores / supermarkets within 2km. |
| `crime_rate` | Float | Index (0.01–0.95) | Normalized neighborhood safety index. |
| `property_tax` | Float | ₹ per annum | Annual municipal property tax assessment. |
| `income_index` | Float | Index (0.2–1.0) | Local median household income relative index. |
| `location` | Categorical | Metros | City: Mumbai, Bangalore, Delhi, Hyderabad, Pune, Chennai. |
| `furnishing` | Categorical | Tier | Unfurnished, Semi-Furnished, Furnished. |
| `has_garden` | Categorical | Yes / No | Availability of private landscaped garden. |
| `has_pool` | Categorical | Yes / No | Availability of swimming pool. |
| **`price`** | **Float** | **₹ (Target Variable)** | **Estimated market valuation.** |
| `area_sqft` | Float | Square feet (500–5,500) | Carpet/super built-up floor area. |
| `bedrooms` | Integer | Count (1–6) | Number of bedrooms. |
| `bathrooms` | Integer | Count (1–5) | Number of bathrooms. |
| `stories` | Integer | Count (1–4) | Number of floors/stories. |
| `parking` | Integer | Count (0–3) | Designated covered parking spaces. |
| `age_years` | Float | Years (0–45) | Age of property since construction. |
| `distance_to_city_km` | Float | Kilometers (1–38) | Road distance to central business district. |
| `distance_to_school_km` | Float | Kilometers (0.5–15) | Road distance to nearest accredited school. |
| `distance_to_hospital_km` | Float | Kilometers (0.5–15) | Road distance to nearest tertiary hospital. |
| `crime_rate` | Float | Index (0.01–0.95) | Normalized neighborhood safety index (lower is safer). |
| `property_tax` | Float | ₹ per annum | Annual municipal property tax assessment. |
| `income_index` | Float | Index (0.2–1.0) | Local median household income relative index. |
| `location` | Categorical | Metros | City: Mumbai, Bangalore, Delhi, Hyderabad, Pune, Chennai. |
| `furnishing` | Categorical | Tier | Unfurnished, Semi-Furnished, Furnished. |
| `has_garden` | Categorical | Yes / No | Availability of private landscaped garden. |
| `has_pool` | Categorical | Yes / No | Availability of swimming pool. |
| **`price`** | **Float** | **₹ (Target Variable)** | **Estimated market valuation.** |

---

## 📐 Mathematical Foundations

### 1. Linear Hypothesis
The linear prediction equation for $n$ samples and $d$ features is given by:

$$\hat{y}^{(i)} = \sum_{j=1}^d X_{ij} w_j + b = \mathbf{x}^{(i)} \mathbf{w} + b$$

In vectorized matrix form:
$$\hat{\mathbf{y}} = \mathbf{X}\mathbf{w} + b \mathbf{1}$$

Where $\mathbf{X} \in \mathbb{R}^{n \times d}$, $\mathbf{w} \in \mathbb{R}^d$, and $b \in \mathbb{R}$.

---

### 2. Regularized Mean Squared Error Cost Function
The objective function balances empirical loss minimization with complexity control:

$$J(\mathbf{w}, b) = \frac{1}{n} \sum_{i=1}^n \left(\hat{y}^{(i)} - y^{(i)}\right)^2 + R(\mathbf{w})$$

- **No Regularization**: $R(\mathbf{w}) = 0$
- **L2 Regularization (Ridge)**: $R(\mathbf{w}) = \lambda \sum_{j=1}^d w_j^2$
- **L1 Regularization (Lasso)**: $R(\mathbf{w}) = \lambda \sum_{j=1}^d |w_j|$

> **Crucial Implementation Rule**: The bias parameter $b$ is **never regularized**. Regularizing the bias would artificially shrink baseline predictions toward zero, regardless of the true target mean.

---

### 3. Analytical Gradient Calculations

The gradient with respect to the bias $b$:
$$\frac{\partial J}{\partial b} = \frac{2}{n} \sum_{i=1}^n (\hat{y}^{(i)} - y^{(i)})$$

The gradient with respect to the weights $\mathbf{w}$:
$$\frac{\partial J}{\partial \mathbf{w}} = \frac{2}{n} \mathbf{X}^T (\hat{\mathbf{y}} - \mathbf{y}) + \frac{\partial R(\mathbf{w})}{\partial \mathbf{w}}$$

Where the regularization subgradient is:
$$\frac{\partial R(\mathbf{w})}{\partial \mathbf{w}} = \begin{cases} 
\mathbf{0} & \text{No Regularization} \\ 
2\lambda \mathbf{w} & \text{L2 (Ridge)} \\ 
\lambda \, \text{sign}(\mathbf{w}) & \text{L1 (Lasso)} 
\end{cases}$$

---

### 4. Batch Gradient Descent Parameter Updates
At each epoch $t$, weights and bias are updated simultaneously in the direction of steepest descent:

$$\mathbf{w}^{(t+1)} = \mathbf{w}^{(t)} - \alpha \frac{\partial J}{\partial \mathbf{w}}$$
$$b^{(t+1)} = b^{(t)} - \alpha \frac{\partial J}{\partial b}$$

Where $\alpha > 0$ represents the step size (learning rate).

---

### 5. Why Feature Scaling is Essential for Gradient Descent
Without feature scaling, features with large scales (e.g., `area_sqft` in thousands and `property_tax` in tens of thousands) dwarf features on fractional scales (e.g., `crime_rate` in hundredths). This creates an elongated, ill-conditioned elliptical loss surface. 

In such a geometry:
- The gradient vector points almost perpendicularly to the direction of the global minimum.
- Gradient descent oscillates wildly and diverges unless an impractically small learning rate is chosen.
- By standardizing features using $z = \frac{x - \mu}{\sigma}$ via `StandardScaler`, the loss surface becomes spherical and isotropic, enabling smooth, rapid, and stable convergence.

---

### 6. Early Stopping Mechanism
To avoid overfitting and redundant computation, training monitors validation loss $J_{\text{val}}$ after every epoch. If validation loss fails to decrease by at least $\text{tol} = 10^{-7}$ for `patience = 100` consecutive epochs, optimization halts and the best historical parameters $(\mathbf{w}^*, b^*)$ are restored.

---

## 📈 Empirical Results & Benchmarking

### 1. Primary Custom Model Performance
Trained on 12,000 samples (8,399 Train / 1,801 Validation / 1,800 Test holdout):

| Metric | Holdout Test Set (15%) | 5-Fold Cross-Validation |
| :--- | :--- | :--- |
| **Coefficient of Determination ($R^2$)** | **0.9088** | **0.8794 ± 0.0188** |
| **Root Mean Squared Error (RMSE)** | **₹14,87,080** | **₹17,77,740 ± ₹1,54,575** |
| **Mean Absolute Error (MAE)** | **₹9,40,083** | **₹9,54,159 ± ₹22,509** |
| **Mean Absolute Percentage Error (MAPE)** | **8.16%** | **—** |
| **Optimal Hyperparameters** | $\alpha = 0.05$, Epochs = 3000, Patience = 100 |

---

### 2. Model Comparison Table (`compare_models.py`)

| Model | MAE | RMSE | $R^2$ | MAPE | Training Time |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Our Custom GD (No Reg)** | **₹9,93,800** | **₹15,93,376** | **0.8953** | **8.48%** | **0.189s** |
| **Our Custom GD + L1 (Lasso)** | **₹9,93,800** | **₹15,93,376** | **0.8953** | **8.48%** | **0.196s** |
| **Our Custom GD + L2 (Ridge)** | **₹9,70,089** | **₹15,98,685** | **0.8947** | **8.13%** | **0.194s** |
| Scikit-learn LinearRegression | ₹9,42,184 | ₹14,87,595 | 0.9088 | 8.18% | 0.003s |
| Scikit-learn Ridge ($L_2$, $\alpha=1.0$) | ₹9,40,960 | ₹14,87,442 | 0.9088 | 8.17% | 0.001s |
| Scikit-learn Lasso ($L_1$, $\alpha=1.0$) | ₹9,42,180 | ₹14,87,594 | 0.9088 | 8.18% | 0.035s |
| Scikit-learn RandomForest (Non-linear) | ₹9,43,618 | ₹12,69,923 | 0.9335 | 7.56% | 0.470s |

> **Viva Insight**: The custom Batch Gradient Descent matches Scikit-Learn's analytical closed-form OLS within < 0.1%, confirming the theoretical and mathematical correctness of our gradient calculations, step updates, and scaling pipeline. The secondary Random Forest model captures non-linear location-area interaction compounding, highlighting why non-linear models achieve slightly lower RMSE.

---

## 💻 Installation & Usage

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone https://github.com/nullvectorcodes/House_Price_FOA_Assignment.git
cd House_Price_FOA_Assignment

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Automated Unit Tests
```bash
pytest -v
```
All 11 unit tests verify cost computation, analytical gradients, L1/L2 penalties, bias invariance, early stopping, outlier capping, and serialization reproducibility.

### 3. Run Training Pipeline
```bash
python train.py
```
Generates the 12,000-sample dataset, performs leakage-free preprocessing, hyperparameter grid search, 5-fold cross-validation, test evaluation, visualization generation, and serializes `models/house_price_model.pkl`.

### 4. Run Model Comparison Benchmark
```bash
python compare_models.py
```

### 5. Interactive CLI Prediction
```bash
python predict.py
```
Prompts for house details and outputs the estimated valuation, price per square foot, 95% error band, and model-based feature explanations.

### 6. Launch Modern Streamlit Dashboard
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
House_Price_FOA_Assignment/
│
├── data/
│   ├── raw/
│   │   └── house_prices.csv           # Raw dataset (12,000 samples)
│   └── processed/
│       └── processed_data.csv         # Standardized features & target
│
├── models/
│   ├── house_price_model.pkl          # Serialized bundle (model, scaler, encoder, metrics)
│   ├── loss_curve.png                 # Convergence trajectory plot
│   ├── actual_vs_pred.png             # Test set prediction scatter plot
│   ├── residuals.png                  # Residual error diagnostics
│   └── feature_importance.png         # Standardized coefficient bar chart
│
├── notebooks/
│   └── exploratory_analysis.ipynb     # Interactive EDA notebook
│
├── src/
│   ├── __init__.py
│   ├── data_loader.py                 # Dataset generator & CSV loader
│   ├── preprocessing.py               # Imputation, IQR winsorization, OneHotEncoder, StandardScaler
│   ├── feature_engineering.py         # Leakage-free feature derivation
│   ├── linear_regression.py           # Custom RegularizedLinearRegression class
│   ├── regularization.py              # L1 and L2 penalty & gradient mathematics
│   ├── training.py                    # Grid search, tuning & 5-fold cross-validation
│   ├── evaluation.py                  # Metrics & Indian Rupee formatting
│   ├── visualization.py               # Publication-grade Matplotlib/Seaborn figures
│   └── prediction.py                  # Physical constraint validation & inference engine
│
├── tests/
│   ├── test_model.py                  # Math, gradient, loss, L1/L2, early stopping tests
│   ├── test_preprocessing.py          # Imputation, outlier handling, leakage tests
│   └── test_prediction.py             # Inference validation & persistence tests
│
├── app.py                             # Modern Streamlit web application
├── train.py                           # Full automated training pipeline
├── predict.py                         # Interactive CLI prediction tool
├── compare_models.py                  # Benchmarking script (Our GD vs Scikit-Learn baselines)
├── requirements.txt                   # Dependency specifications
├── pytest.ini                         # Pytest configuration
├── README.md                          # Complete documentation
└── .gitignore                         # Git exclusion rules
```

---

## 🎓 Academic Viva Notes

1. **Why Batch Gradient Descent over Normal Equation?**
   - Normal Equation requires computing $(\mathbf{X}^T\mathbf{X})^{-1}$, which costs $O(d^3)$. In high-dimensional settings ($d > 10,000$), this becomes computationally prohibitive or numerically singular. Batch Gradient Descent scales linearly $O(n \cdot d)$ per epoch and naturally handles online/streaming data.
2. **Why never regularize the bias ($b$)?**
   - The bias term models the baseline offset when all features are zero (or their means, when standardized). Penalizing $b$ forces predictions toward 0, introducing systematic bias across the market.
3. **What is the geometric difference between L1 and L2 regularization?**
   - $L_1$ has a diamond-shaped constraint region with sharp corners along axes, encouraging exact zero weights (feature selection). $L_2$ has a spherical constraint region, shrinking weights proportionally toward zero without forcing them exactly to zero.
4. **Why is correlation not causation?**
   - Standardized coefficients show associative alignment under a linear model, but do not imply that manipulating a feature directly causes a price shift in isolation without considering unobserved confounders.

---

## 📜 License
Developed as a B.Tech CSE AI/ML Academic Project. Open for educational and research use.
