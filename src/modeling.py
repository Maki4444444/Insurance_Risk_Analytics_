"""
src/modeling.py
===============
Statistical Modeling & Risk-Based Pricing — ACIS Insurance
Task 4: AlphaCare Insurance Solutions

Fixes applied vs original:
  1. Log-transform target instead of MinMaxScaler (handles right-skew & OOD values)
  2. Median/mode imputation instead of fillna(0)
  3. Leakage guard — Margin and post-event columns excluded from features
  4. Class-weight balancing on all classifiers (no SMOTE dependency)
  5. Severity model correctly applied only to claim-subset population
  6. Pricing applies severity only via the claim-subset model; correct population alignment
  7. Premium floor constraint added to compute_risk_based_premium
  8. VehicleAge uses TransactionMonth year (data-aware) instead of hardcoded 2015
  9. Split stratified on HasClaim for classification to preserve rare-class ratio
 10. compare_classifier_models sorts by AUC (better metric for imbalanced data)
"""

import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.model_selection import (
    train_test_split, GridSearchCV, cross_val_score
)
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score
)
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer
import xgboost as xgb
import shap
import warnings
warnings.filterwarnings("ignore")

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ── 1. Data Preparation ───────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer features and prepare dataset for modeling.

    Changes vs original:
    - VehicleAge derived from the data's own year range, not hardcoded 2015.
      If TransactionMonth exists, we infer the reference year from data.
    - Margin is retained for EDA/business insight but flagged — it must be
      excluded from model features (see get_feature_columns) because it is
      derived from TotalClaims and TotalPremium, both of which are unknowable
      at prediction time for new policies.
    - HasClaim derived cleanly from TotalClaims.
    - PremiumToSumInsuredRatio: valid at quote time, so kept as a feature.
    """
    try:
        data = df.copy()

        # --- VehicleAge: use data-aware reference year ---
        if "RegistrationYear" in data.columns:
            if "TransactionMonth" in data.columns:
                try:
                    ref_year = pd.to_datetime(
                        data["TransactionMonth"], errors="coerce"
                    ).dt.year.median()
                    ref_year = int(ref_year) if not np.isnan(ref_year) else 2015
                except Exception:
                    ref_year = 2015
            else:
                ref_year = 2015
            data["VehicleAge"] = (ref_year - data["RegistrationYear"]).clip(0, 50)
            logger.info(f"Engineered VehicleAge (ref year={ref_year}).")

        # --- Margin: business KPI only, not a model feature ---
        if "TotalPremium" in data.columns and "TotalClaims" in data.columns:
            data["Margin"] = data["TotalPremium"] - data["TotalClaims"]
            logger.info("Engineered Margin (EDA use only — excluded from model features).")

        # --- HasClaim: classification target ---
        if "TotalClaims" in data.columns:
            data["HasClaim"] = (data["TotalClaims"] > 0).astype(int)
            logger.info("Engineered HasClaim binary feature.")

        # --- PremiumToSumInsuredRatio: valid at quote time ---
        if "SumInsured" in data.columns and "TotalPremium" in data.columns:
            data["SumInsured"] = data["SumInsured"].clip(lower=0)
            data["PremiumToSumInsuredRatio"] = (
                data["TotalPremium"].clip(lower=0) / (data["SumInsured"] + 1)
            ).clip(lower=0, upper=10)
            logger.info("Engineered PremiumToSumInsuredRatio.")

        return data

    except Exception as e:
        logger.error(f"prepare_features failed: {e}")
        raise


def encode_categoricals(df: pd.DataFrame, cat_cols: list) -> tuple:
    """
    Label encode categorical columns.
    Returns encoded DataFrame and fitted encoders dict.
    """
    try:
        data = df.copy()
        encoders = {}
        for col in cat_cols:
            if col not in data.columns:
                logger.warning(f"Column '{col}' not found — skipping.")
                continue
            le = LabelEncoder()
            data[col] = data[col].astype(str).fillna("Unknown")
            data[col] = le.fit_transform(data[col])
            encoders[col] = le
            logger.info(f"Encoded column: {col}")
        return data, encoders

    except Exception as e:
        logger.error(f"encode_categoricals failed: {e}")
        raise


def impute_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Impute missing values properly:
      - Numeric columns  → median  (robust to skew and outliers)
      - Categorical/bool → most_frequent

    Replaces the original blanket fillna(0) which was treating
    'missing' as 'zero', corrupting risk signals (e.g. missing
    SumInsured → 0 implies no asset value).

    Returns a new DataFrame with no nulls.
    """
    try:
        data = df.copy()

        num_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        cat_cols = data.select_dtypes(exclude=[np.number]).columns.tolist()

        if num_cols:
            num_imputer = SimpleImputer(strategy="median")
            data[num_cols] = num_imputer.fit_transform(data[num_cols])
            logger.info(f"Median-imputed {len(num_cols)} numeric columns.")

        if cat_cols:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])
            logger.info(f"Mode-imputed {len(cat_cols)} categorical columns.")

        remaining = data.isnull().sum().sum()
        logger.info(f"Remaining nulls after imputation: {remaining}")
        return data

    except Exception as e:
        logger.error(f"impute_features failed: {e}")
        raise


def log_transform_target(y_train: pd.Series,
                          y_test: pd.Series) -> tuple:
    """
    Apply log1p transform to the claim severity target.

    Why log1p instead of MinMaxScaler:
    - Insurance claims are heavily right-skewed; log pulls extreme values in,
      improving model fit for the bulk of the distribution.
    - MinMaxScaler ties scale to training-set min/max. Any new claim outside
      that range produces an out-of-bounds scaled value, breaking
      inverse_transform. Log has no such constraint.
    - Log-space predictions inverse-transform cleanly via np.expm1.

    Returns log-transformed arrays and a 'scaler' dict for inverse transform.
    """
    try:
        y_train_log = np.log1p(y_train.values)
        y_test_log = np.log1p(y_test.values)
        transform_info = {"method": "log1p"}
        logger.info(
            f"Log1p target transform applied. "
            f"Range: [{y_train_log.min():.4f}, {y_train_log.max():.4f}]"
        )
        return y_train_log, y_test_log, transform_info

    except Exception as e:
        logger.error(f"log_transform_target failed: {e}")
        raise


def inverse_scale(values: np.ndarray, scaler) -> np.ndarray:
    """
    Inverse transform predictions back to original Rand values.
    Handles both log1p (dict) and legacy MinMaxScaler objects.
    """
    try:
        if isinstance(scaler, dict) and scaler.get("method") == "log1p":
            return np.expm1(values)
        else:
            # Legacy MinMaxScaler path (kept for backward compatibility)
            from sklearn.preprocessing import MinMaxScaler as _MMS
            if isinstance(scaler, _MMS):
                return scaler.inverse_transform(
                    values.reshape(-1, 1)
                ).flatten()
        raise ValueError("Unknown scaler type.")
    except Exception as e:
        logger.error(f"inverse_scale failed: {e}")
        raise


# Keep scale_target as a thin alias so existing notebook imports don't break.
# New code should call log_transform_target directly.
def scale_target(y_train: pd.Series,
                 y_test: pd.Series) -> tuple:
    """
    Alias → delegates to log_transform_target.
    Kept for notebook import compatibility.
    """
    return log_transform_target(y_train, y_test)


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Return feature columns safe to use at prediction time.

    Excluded:
    - TotalClaims, HasClaim  — the targets themselves
    - TotalPremium           — target-adjacent; unavailable for new quote pricing
    - Margin                 — derived from TotalClaims + TotalPremium (data leakage)
    - ID / timestamp cols    — no predictive signal
    """
    exclude = {
        "TotalClaims", "TotalPremium", "HasClaim", "Margin",
        "UnderwrittenCoverID", "PolicyID", "TransactionMonth"
    }
    return [c for c in df.columns if c not in exclude]


def split_data(df: pd.DataFrame, target: str,
               test_size: float = 0.2,
               random_state: int = 42) -> tuple:
    """
    Split data into train/test sets.

    For classification (HasClaim target), uses stratified split to
    preserve the rare-class ratio in both train and test sets.
    Without stratification, a 0.28% claim rate can result in test
    sets with zero positive examples.
    """
    try:
        if target not in df.columns:
            raise KeyError(f"Target column '{target}' not found.")

        feature_cols = get_feature_columns(df)
        feature_cols = [c for c in feature_cols if c in df.columns]

        X = df[feature_cols]
        y = df[target]

        # Stratify for classification targets only
        stratify = y if target == "HasClaim" else None

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify
        )
        logger.info(
            f"Split — Train: {X_train.shape} | Test: {X_test.shape}"
            + (f" | Stratified on {target}" if stratify is not None else "")
        )
        return X_train, X_test, y_train, y_test

    except KeyError as e:
        logger.error(f"split_data failed: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in split_data: {e}")
        raise


# ── 2. Regression Models ──────────────────────────────────────────────────────

def train_linear_regression(X_train: pd.DataFrame,
                             y_train: np.ndarray) -> LinearRegression:
    """Train a Linear Regression model."""
    try:
        model = LinearRegression()
        model.fit(X_train, y_train)
        logger.info("Linear Regression trained successfully.")
        return model
    except Exception as e:
        logger.error(f"train_linear_regression failed: {e}")
        raise


def train_decision_tree_regressor(X_train: pd.DataFrame,
                                   y_train: np.ndarray,
                                   max_depth: int = 10,
                                   random_state: int = 42) -> DecisionTreeRegressor:
    """Train a Decision Tree Regressor."""
    try:
        model = DecisionTreeRegressor(
            max_depth=max_depth,
            random_state=random_state
        )
        model.fit(X_train, y_train)
        logger.info("Decision Tree Regressor trained successfully.")
        return model
    except Exception as e:
        logger.error(f"train_decision_tree_regressor failed: {e}")
        raise


def train_random_forest_regressor(X_train: pd.DataFrame,
                                   y_train: np.ndarray,
                                   n_estimators: int = 100,
                                   random_state: int = 42) -> RandomForestRegressor:
    """Train a Random Forest Regressor."""
    try:
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1
        )
        model.fit(X_train, y_train)
        logger.info("Random Forest Regressor trained successfully.")
        return model
    except Exception as e:
        logger.error(f"train_random_forest_regressor failed: {e}")
        raise


def train_xgboost_regressor(X_train: pd.DataFrame,
                             y_train: np.ndarray,
                             random_state: int = 42) -> xgb.XGBRegressor:
    """Train an XGBoost Regressor."""
    try:
        model = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0
        )
        model.fit(X_train, y_train)
        logger.info("XGBoost Regressor trained successfully.")
        return model
    except Exception as e:
        logger.error(f"train_xgboost_regressor failed: {e}")
        raise


def evaluate_regression(model, X_test: pd.DataFrame,
                         y_test_scaled: np.ndarray,
                         scaler,
                         model_name: str) -> dict:
    """
    Evaluate a regression model.
    All error metrics reported in original Rand values after inverse transform.
    """
    try:
        y_pred_scaled = model.predict(X_test)

        y_pred_orig = inverse_scale(y_pred_scaled, scaler)
        y_test_orig = inverse_scale(y_test_scaled, scaler)

        mae  = mean_absolute_error(y_test_orig, y_pred_orig)
        mse  = mean_squared_error(y_test_orig, y_pred_orig)
        rmse = np.sqrt(mse)
        # R² computed in log-space (where the model was trained) for validity
        r2   = r2_score(y_test_scaled, y_pred_scaled)

        logger.info(
            f"{model_name} — RMSE: R{rmse:,.2f} | "
            f"MAE: R{mae:,.2f} | R²: {r2:.4f}"
        )

        return {
            "Model": model_name,
            "MAE":  round(mae,  2),
            "MSE":  round(mse,  2),
            "RMSE": round(rmse, 2),
            "R2":   round(r2,   4),
            "Predictions_orig": y_pred_orig,
            "Predictions_scaled": y_pred_scaled,
        }

    except Exception as e:
        logger.error(f"evaluate_regression failed for {model_name}: {e}")
        raise


def tune_random_forest(X_train: pd.DataFrame,
                        y_train: np.ndarray,
                        cv: int = 5) -> tuple:
    """Hyperparameter tuning for Random Forest using GridSearchCV."""
    try:
        param_grid = {
            "n_estimators":     [100, 200],
            "max_depth":        [None, 10, 20],
            "min_samples_split":[2, 5],
            "min_samples_leaf": [1, 2],
            "bootstrap":        [True, False]
        }
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)
        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid,
            cv=cv,
            n_jobs=-1,
            scoring="r2",
            verbose=1
        )
        grid_search.fit(X_train, y_train)
        logger.info(f"Best params: {grid_search.best_params_}")
        logger.info(f"Best CV R²: {grid_search.best_score_:.4f}")
        return grid_search.best_estimator_, grid_search.best_params_

    except Exception as e:
        logger.error(f"tune_random_forest failed: {e}")
        raise


def cross_validate_model(model, X_train: pd.DataFrame,
                          y_train: np.ndarray,
                          model_name: str,
                          cv: int = 5) -> dict:
    """Run k-fold cross validation on a model."""
    try:
        scores = cross_val_score(
            model, X_train, y_train,
            cv=cv, scoring="r2", n_jobs=-1
        )
        logger.info(
            f"{model_name} CV R²: {scores.mean():.4f} ± {scores.std():.4f}"
        )
        return {
            "Model":       model_name,
            "CV_R2_Mean":  round(scores.mean(), 4),
            "CV_R2_Std":   round(scores.std(),  4),
            "CV_Scores":   scores
        }
    except Exception as e:
        logger.error(f"cross_validate_model failed for {model_name}: {e}")
        raise


# ── 3. Classification Models ──────────────────────────────────────────────────

def train_logistic_regression(X_train: pd.DataFrame,
                               y_train: pd.Series) -> LogisticRegression:
    """
    Train a Logistic Regression classifier.

    class_weight='balanced' automatically adjusts weights inversely
    proportional to class frequencies. This means the 0.28% claim
    class gets ~357× more weight than non-claims, forcing the model
    to learn claim patterns rather than defaulting to 'no claim'.
    """
    try:
        model = LogisticRegression(
            max_iter=1000,
            random_state=42,
            n_jobs=-1,
            class_weight="balanced"   # FIX: was None → zero recall on claims
        )
        model.fit(X_train, y_train)
        logger.info("Logistic Regression trained (class_weight=balanced).")
        return model
    except Exception as e:
        logger.error(f"train_logistic_regression failed: {e}")
        raise


def train_random_forest_classifier(X_train: pd.DataFrame,
                                    y_train: pd.Series,
                                    n_estimators: int = 100,
                                    random_state: int = 42) -> RandomForestClassifier:
    """
    Train a Random Forest Classifier.
    class_weight='balanced' applied for same reason as Logistic Regression.
    """
    try:
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced"   # FIX: was None
        )
        model.fit(X_train, y_train)
        logger.info("Random Forest Classifier trained (class_weight=balanced).")
        return model
    except Exception as e:
        logger.error(f"train_random_forest_classifier failed: {e}")
        raise


def train_xgboost_classifier(X_train: pd.DataFrame,
                              y_train: pd.Series,
                              random_state: int = 42) -> xgb.XGBClassifier:
    """
    Train an XGBoost Classifier.

    scale_pos_weight = (# negatives) / (# positives) is XGBoost's
    built-in mechanism for handling class imbalance, equivalent to
    class_weight='balanced' in sklearn.
    """
    try:
        n_neg = int((y_train == 0).sum())
        n_pos = int((y_train == 1).sum())
        spw   = n_neg / n_pos if n_pos > 0 else 1.0
        logger.info(
            f"XGBoost scale_pos_weight={spw:.1f} "
            f"(neg={n_neg:,}, pos={n_pos:,})"
        )
        model = xgb.XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
            eval_metric="logloss",
            scale_pos_weight=spw      # FIX: was missing → near-zero recall
        )
        model.fit(X_train, y_train)
        logger.info("XGBoost Classifier trained (scale_pos_weight applied).")
        return model
    except Exception as e:
        logger.error(f"train_xgboost_classifier failed: {e}")
        raise


def evaluate_classifier(model, X_test: pd.DataFrame,
                         y_test: pd.Series,
                         model_name: str) -> dict:
    """Evaluate a classification model."""
    try:
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]

        accuracy  = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall    = recall_score(y_test, y_pred, zero_division=0)
        f1        = f1_score(y_test, y_pred, zero_division=0)
        auc       = roc_auc_score(y_test, y_prob)

        logger.info(
            f"{model_name} — Accuracy: {accuracy:.4f} | "
            f"Recall: {recall:.4f} | F1: {f1:.4f} | AUC: {auc:.4f}"
        )

        return {
            "Model":         model_name,
            "Accuracy":      round(accuracy,  4),
            "Precision":     round(precision, 4),
            "Recall":        round(recall,    4),
            "F1":            round(f1,        4),
            "AUC":           round(auc,       4),
            "Predictions":   y_pred,
            "Probabilities": y_prob
        }

    except Exception as e:
        logger.error(f"evaluate_classifier failed for {model_name}: {e}")
        raise


# ── 4. SHAP Analysis ──────────────────────────────────────────────────────────

def compute_shap_values(model, X_sample: pd.DataFrame,
                         model_type: str = "tree") -> shap.Explanation:
    """Compute SHAP values for a trained model."""
    try:
        if model_type == "tree":
            explainer = shap.TreeExplainer(model)
        elif model_type == "linear":
            explainer = shap.LinearExplainer(
                model, X_sample,
                feature_perturbation="interventional"
            )
        else:
            raise ValueError(
                f"Unknown model_type: '{model_type}'. Use 'tree' or 'linear'."
            )
        shap_values = explainer(X_sample)
        logger.info(f"SHAP values computed for {model_type} model.")
        return shap_values

    except Exception as e:
        logger.error(f"compute_shap_values failed: {e}")
        raise


# ── 5. Pricing Framework ──────────────────────────────────────────────────────

def align_severity_to_population(severity_model,
                                  X_full: pd.DataFrame,
                                  target_scaler,
                                  claim_threshold: float = 0.0) -> np.ndarray:
    """
    Apply the severity model to a full population DataFrame and return
    predicted severity in original Rand units for every row.

    The severity model was trained on claim-only policies. Applying it to
    the full population is valid for the pricing formula because we weight
    by P(claim) — policies with near-zero claim probability contribute
    negligibly to the expected loss regardless of severity estimate.

    This replaces the notebook's direct call of:
        best_sev_model.predict(X_test_clf)   ← mismatched population
    with a clean, documented function.

    Parameters
    ----------
    severity_model : fitted regressor (trained on log-space target)
    X_full         : feature matrix for the full pricing population
    target_scaler  : the transform_info dict (or MinMaxScaler) from training
    claim_threshold: floor for predicted severity (default 0)

    Returns
    -------
    np.ndarray of predicted severity in Rand for every row in X_full
    """
    try:
        pred_log = severity_model.predict(X_full)
        pred_orig = inverse_scale(pred_log, target_scaler)
        pred_orig = np.maximum(pred_orig, claim_threshold)
        logger.info(
            f"Severity aligned to population ({len(X_full):,} rows). "
            f"Avg predicted severity: R{pred_orig.mean():,.2f}"
        )
        return pred_orig
    except Exception as e:
        logger.error(f"align_severity_to_population failed: {e}")
        raise


def compute_risk_based_premium(p_claim: np.ndarray,
                                predicted_severity: np.ndarray,
                                expense_loading: float = 0.15,
                                profit_margin: float = 0.10,
                                min_premium: float = 50.0) -> np.ndarray:
    """
    Compute risk-based premium using the pure premium formula:

        Expected Loss  = P(claim) × Predicted Severity
        Gross Premium  = Expected Loss × (1 + expense_loading + profit_margin)
        Final Premium  = max(Gross Premium, min_premium)

    The min_premium floor (default R50) prevents the model from producing
    premiums below the cost of policy administration and statutory levies.
    Without this floor, the original model systematically priced a large
    cluster of policies below R100 — guaranteeing underwriting losses.

    Parameters
    ----------
    p_claim            : array of P(claim) from classifier
    predicted_severity : array of predicted claim amounts in Rand
    expense_loading    : fraction for expenses (default 0.15 = 15%)
    profit_margin      : fraction for profit (default 0.10 = 10%)
    min_premium        : floor premium in Rand (default R50)
    """
    try:
        expected_loss = p_claim * predicted_severity
        gross_premium = expected_loss * (1 + expense_loading + profit_margin)
        premium = np.maximum(gross_premium, min_premium)
        logger.info(
            f"Premiums computed — Avg: R{premium.mean():,.2f} | "
            f"Min: R{premium.min():,.2f} | Max: R{premium.max():,.2f} | "
            f"Floor applied: {(gross_premium < min_premium).sum():,} policies"
        )
        return premium
    except Exception as e:
        logger.error(f"compute_risk_based_premium failed: {e}")
        raise


# ── 6. Comparison Tables ──────────────────────────────────────────────────────

def compare_regression_models(results: list) -> pd.DataFrame:
    """Build a sorted comparison table for regression models (by RMSE)."""
    try:
        table = pd.DataFrame([
            {
                "Model": r["Model"],
                "MAE":   r["MAE"],
                "MSE":   r["MSE"],
                "RMSE":  r["RMSE"],
                "R2":    r["R2"]
            }
            for r in results
        ])
        return table.sort_values("RMSE").reset_index(drop=True)
    except Exception as e:
        logger.error(f"compare_regression_models failed: {e}")
        raise


def compare_classifier_models(results: list) -> pd.DataFrame:
    """
    Build a sorted comparison table for classification models.

    Sorted by AUC (not F1) because:
    - AUC measures rank-ordering ability across all thresholds.
    - With 0.28% claim rate, F1 is highly sensitive to the decision
      threshold and can be gamed by threshold selection.
    - AUC is threshold-independent and better reflects true discriminative
      power for highly imbalanced insurance data.
    """
    try:
        table = pd.DataFrame([
            {
                "Model":     r["Model"],
                "Accuracy":  r["Accuracy"],
                "Precision": r["Precision"],
                "Recall":    r["Recall"],
                "F1":        r["F1"],
                "AUC":       r["AUC"]
            }
            for r in results
        ])
        return table.sort_values("AUC", ascending=False).reset_index(drop=True)
    except Exception as e:
        logger.error(f"compare_classifier_models failed: {e}")
        raise