"""
End-to-End Training Pipeline for House Price Prediction.

Workflow:
1. Generates or loads raw dataset (>= 10,000 samples).
2. Performs 70% Train / 15% Validation / 15% Test split (random_state=42).
3. Fits Preprocessing pipeline strictly on training data.
4. Performs Hyperparameter Tuning across learning rates and L1/L2 penalties.
5. Trains final model with Early Stopping (patience=100) and saves loss convergence.
6. Runs 5-Fold Cross-Validation.
7. Evaluates on unseen Test Set.
8. Generates visualization artifacts.
9. Serializes complete model bundle to models/house_price_model.pkl.
"""

from pathlib import Path
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import generate_synthetic_data, load_data
from src.evaluation import calculate_metrics, format_inr
from src.linear_regression import RegularizedLinearRegression
from src.preprocessing import DataPreprocessor, TARGET_COL
from src.training import k_fold_cross_validation, tune_hyperparameters
from src.visualization import (
    plot_actual_vs_predicted,
    plot_feature_importance,
    plot_residuals,
    plot_training_loss,
)


# Directories
DATA_RAW_DIR = Path("data/raw")
DATA_PROC_DIR = Path("data/processed")
MODELS_DIR = Path("models")

RAW_DATA_PATH = DATA_RAW_DIR / "house_prices.csv"
PROC_DATA_PATH = DATA_PROC_DIR / "processed_data.csv"
MODEL_BUNDLE_PATH = MODELS_DIR / "house_price_model.pkl"


def main() -> None:
    start_time = time.time()
    print("=" * 70)
    print(" 🚀 STARTING HOUSE PRICE PREDICTION TRAINING PIPELINE")
    print("=" * 70)

    # Ensure directories exist
    DATA_RAW_DIR.mkdir(parents=True, exist_ok=True)
    DATA_PROC_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Dataset Generation or Loading
    if not RAW_DATA_PATH.exists():
        print(f"\n[1/7] Generating synthetic housing dataset (12,000 samples)...")
        df_raw = generate_synthetic_data(n_samples=12000, random_state=42)
        df_raw.to_csv(RAW_DATA_PATH, index=False)
        print(f"      Saved raw dataset to: {RAW_DATA_PATH}")
    else:
        print(f"\n[1/7] Loading existing raw dataset from: {RAW_DATA_PATH}...")
        df_raw = load_data(RAW_DATA_PATH)

    print(f"      Dataset Shape: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")

    # Check and remove duplicates
    init_rows = len(df_raw)
    df_raw = df_raw.drop_duplicates().reset_index(drop=True)
    dups_removed = init_rows - len(df_raw)
    if dups_removed > 0:
        print(f"      Detected and removed {dups_removed} duplicate rows.")

    # 2. Train / Validation / Test Split (70% Train, 15% Val, 15% Test)
    print("\n[2/7] Splitting dataset into Train (70%), Validation (15%), Test (15%)...")
    df_train_val, df_test = train_test_split(
        df_raw, test_size=0.15, random_state=42, shuffle=True
    )
    # Remaining 85%: 15/85 ≈ 0.17647 for validation to get exactly 15% overall
    val_fraction = 0.15 / 0.85
    df_train, df_val = train_test_split(
        df_train_val, test_size=val_fraction, random_state=42, shuffle=True
    )

    print(f"      Training samples:   {len(df_train):,} ({len(df_train)/len(df_raw)*100:.1f}%)")
    print(f"      Validation samples: {len(df_val):,} ({len(df_val)/len(df_raw)*100:.1f}%)")
    print(f"      Testing samples:    {len(df_test):,} ({len(df_test)/len(df_raw)*100:.1f}%)")

    y_train = df_train[TARGET_COL].values
    y_val = df_val[TARGET_COL].values
    y_test = df_test[TARGET_COL].values

    # 3. Preprocessing (Fitted strictly on df_train)
    print("\n[3/7] Fitting Preprocessing Pipeline (Leakage Prevention)...")
    preprocessor = DataPreprocessor(iqr_multiplier=1.5)
    X_train = preprocessor.fit_transform(df_train)
    X_val = preprocessor.transform(df_val)
    X_test = preprocessor.transform(df_test)

    feature_names = preprocessor.feature_names_
    print(f"      Transformed features count: {len(feature_names)}")
    print(f"      Top features: {feature_names[:6]} ...")

    # Save processed dataset sample
    df_processed_sample = pd.DataFrame(X_train, columns=feature_names)
    df_processed_sample["price"] = y_train
    df_processed_sample.to_csv(PROC_DATA_PATH, index=False)
    print(f"      Saved processed dataset to: {PROC_DATA_PATH}")

    # 4. Hyperparameter Tuning
    print("\n[4/7] Running Hyperparameter Grid Search on Validation Set...")
    print("      Testing Learning Rates: [0.001, 0.005, 0.01, 0.05]")
    print("      Testing Penalties:      [None, L1, L2]")
    print("      Testing Lambdas:        [0, 0.001, 0.01, 0.1, 1.0]")

    best_params, tuning_df = tune_hyperparameters(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        learning_rates=[0.001, 0.005, 0.01, 0.05],
        lambdas=[0.0, 0.001, 0.01, 0.1, 1.0],
        penalties=[None, "l1", "l2"],
        n_epochs=2000,
        patience=100,
    )

    print("\n      Top 5 Hyperparameter Combinations (Ranked by Val RMSE):")
    print(tuning_df.head(5).to_string(index=False))

    print(f"\n      Selected Best Parameters:")
    print(f"        • Penalty:        {best_params['penalty']}")
    print(f"        • Learning Rate:  {best_params['learning_rate']}")
    print(f"        • Lambda:         {best_params['lmbda']}")
    print(f"        • Validation RMSE:{format_inr(best_params['val_rmse'])}")

    # 5. Train Final Model with Early Stopping
    print("\n[5/7] Training Final Model with Early Stopping (patience=100)...")
    final_model = RegularizedLinearRegression(
        learning_rate=best_params["learning_rate"],
        n_epochs=3000,
        penalty=best_params["penalty"],
        lmbda=best_params["lmbda"],
        patience=100,
    )

    final_model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    print(f"      Training Completed:")
    print(f"        • Best Epoch:          {final_model.best_epoch}")
    print(f"        • Final Epoch:         {final_model.final_epoch}")
    print(f"        • Early Stopped:       {final_model.early_stopped}")
    print(f"        • Best Validation Loss:{final_model.best_val_loss:.4e}")

    # 6. 5-Fold Cross Validation
    print("\n[6/7] Running 5-Fold Cross-Validation...")
    X_full_train = np.vstack([X_train, X_val])
    y_full_train = np.concatenate([y_train, y_val])

    cv_results = k_fold_cross_validation(
        X=X_full_train,
        y=y_full_train,
        k=5,
        learning_rate=best_params["learning_rate"],
        penalty=best_params["penalty"],
        lmbda=best_params["lmbda"],
        n_epochs=2000,
    )

    print(f"      CV Mean RMSE: {format_inr(cv_results['mean_rmse'])} (± {format_inr(cv_results['std_rmse'])})")
    print(f"      CV Mean MAE:  {format_inr(cv_results['mean_mae'])} (± {format_inr(cv_results['std_mae'])})")
    print(f"      CV Mean R²:   {cv_results['mean_r2']:.4f} (± {cv_results['std_r2']:.4f})")

    # 7. Final Test Set Evaluation
    print("\n[7/7] Evaluating on Unseen Test Set (15% holdout)...")
    y_pred_test = final_model.predict(X_test)
    test_metrics = calculate_metrics(y_test, y_pred_test)

    print("\n" + "=" * 55)
    print(" 🏆 FINAL TEST SET EVALUATION METRICS")
    print("=" * 55)
    print(f"  R² Score:                       {test_metrics['r2']:.4f}")
    print(f"  RMSE (Root Mean Squared Error): {format_inr(test_metrics['rmse'])}")
    print(f"  MAE (Mean Absolute Error):      {format_inr(test_metrics['mae'])}")
    print(f"  MSE (Mean Squared Error):       {test_metrics['mse']:,.0f}")
    print(f"  MAPE (Mean Abs % Error):        {test_metrics['mape']:.2f}%")
    print("=" * 55)

    # Visualizations
    print("\nSaving evaluation plots...")
    plot_training_loss(final_model.history, final_model.best_epoch, save_path=str(MODELS_DIR / "loss_curve.png"))
    plot_actual_vs_predicted(y_test, y_pred_test, save_path=str(MODELS_DIR / "actual_vs_pred.png"))
    plot_residuals(y_test, y_pred_test, save_path=str(MODELS_DIR / "residuals.png"))
    plot_feature_importance(feature_names, final_model.weights, save_path=str(MODELS_DIR / "feature_importance.png"))
    print("Saved plots to models/ directory.")

    # Save complete bundle
    model_bundle = {
        "model": final_model,
        "preprocessor": preprocessor,
        "feature_names": feature_names,
        "test_metrics": test_metrics,
        "cv_results": cv_results,
        "best_params": best_params,
        "tuning_history": tuning_df,
        "history": final_model.history,
        "dataset_stats": {
            "total_samples": len(df_raw),
            "n_features": len(feature_names),
            "train_samples": len(df_train),
            "val_samples": len(df_val),
            "test_samples": len(df_test),
        },
    }

    joblib.dump(model_bundle, MODEL_BUNDLE_PATH)
    print(f"\n✅ Serialized complete model bundle to: {MODEL_BUNDLE_PATH}")
    print(f"⏱️ Total training pipeline runtime: {time.time() - start_time:.2f} seconds.")
    print("=" * 70)


if __name__ == "__main__":
    main()
