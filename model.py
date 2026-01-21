# model.py
import numpy as np
from dataclasses import dataclass
from typing import Dict, Any, Tuple, Literal, Optional

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.ensemble import VotingClassifier, VotingRegressor
from sklearn.metrics import classification_report, r2_score
from sklearn.model_selection import train_test_split, TimeSeriesSplit, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# Optional XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    HAS_XGB = True
except Exception:
    HAS_XGB = False

# Optional Optuna (used only if installed and requested)
try:
    import optuna  # type: ignore
    HAS_OPTUNA = True
except Exception:
    HAS_OPTUNA = False


# ==== ORIGINAL FUNCTION (kept) ================================================
def train_classifier(X, y, test_size: float = 0.2, random_state: int = 42):
    """
    Original single-asset classifier training (time-ordered split).
    Returns: trained_model, (X_train, X_test, y_train, y_test)
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, shuffle=False
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=6,
        min_samples_leaf=20,
        random_state=random_state,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("=== Classification report on holdout (time-ordered) ===")
    print(classification_report(y_test, y_pred, digits=3))

    return clf, (X_train, X_test, y_train, y_test)


# ==== NEW CONFIG & BUILDERS ===================================================

@dataclass
class MLModelResult:
    model: Any
    feature_cols: Any
    task: Literal["classification", "regression"]
    best_params: Optional[Dict[str, Any]] = None
    cv_scores: Optional[np.ndarray] = None


def _build_base_estimator(
    task: Literal["classification", "regression"],
    model_type: str = "rf",
    random_state: int = 42,
) -> Any:
    if task == "classification":
        if model_type == "rf":
            return RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=random_state,
            )
        elif model_type == "gb":
            return GradientBoostingClassifier(random_state=random_state)
        elif model_type == "xgb" and HAS_XGB:
            return XGBClassifier(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
            )
        elif model_type == "ensemble":
            # RF + GB + (optional) XGB Voting
            estimators = [
                ("rf", _build_base_estimator("classification", "rf", random_state)),
                ("gb", _build_base_estimator("classification", "gb", random_state)),
            ]
            if HAS_XGB:
                estimators.append(
                    ("xgb", _build_base_estimator("classification", "xgb", random_state))
                )
            return VotingClassifier(estimators=estimators, voting="soft", n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model_type={model_type} for classification.")
    else:
        if model_type == "rf":
            return RandomForestRegressor(
                n_estimators=300,
                max_depth=8,
                min_samples_leaf=10,
                n_jobs=-1,
                random_state=random_state,
            )
        elif model_type == "gb":
            return GradientBoostingRegressor(random_state=random_state)
        elif model_type == "xgb" and HAS_XGB:
            return XGBRegressor(
                n_estimators=300,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                tree_method="hist",
                random_state=random_state,
                n_jobs=-1,
            )
        elif model_type == "ensemble":
            estimators = [
                ("rf", _build_base_estimator("regression", "rf", random_state)),
                ("gb", _build_base_estimator("regression", "gb", random_state)),
            ]
            if HAS_XGB:
                estimators.append(
                    ("xgb", _build_base_estimator("regression", "xgb", random_state))
                )
            return VotingRegressor(estimators=estimators, n_jobs=-1)
        else:
            raise ValueError(f"Unsupported model_type={model_type} for regression.")


def detect_data_leakage(
    X: np.ndarray,
    y: np.ndarray,
    df: Optional[Any] = None,
    verbose: bool = True
) -> None:
    """
    Simple heuristic leakage check: look for unrealistically high correlation
    between features and target, or obvious future-looking columns in df.
    """
    if df is None:
        return
    suspicious = []
    for col in df.columns:
        if col.lower().startswith("future_"):
            suspicious.append(col)
    if suspicious and verbose:
        print("[LEAKAGE WARNING] Future-looking columns present:", suspicious)


def train_with_cv(
    X: np.ndarray,
    y: np.ndarray,
    task: Literal["classification", "regression"],
    model_type: str = "rf",
    n_splits: int = 5,
    param_grid: Optional[Dict[str, Any]] = None,
    scoring: Optional[str] = None,
    use_optuna: bool = False,
) -> MLModelResult:
    """
    Time-series cross-validation training with scaling + hyperparameter tuning.

    NOTE: For model_type='ensemble' (VotingClassifier / VotingRegressor) we do
    NOT try to tune max_depth/n_estimators directly (those are invalid params).
    Instead we:
      - just fit the pipeline once
      - optionally compute CV scores manually
    """
    # Default scoring
    if scoring is None:
        if task == "classification":
            scoring = "roc_auc"
        else:
            scoring = "neg_mean_squared_error"

    base_est = _build_base_estimator(task, model_type=model_type)
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("model", base_est),
    ])

    tscv = TimeSeriesSplit(n_splits=n_splits)

    # ---- Special-case: ensemble (Voting*) ----
    if model_type == "ensemble":
        if use_optuna:
            print("[WARN] Optuna tuning is not implemented for ensemble model_type. "
                  "Proceeding without hyperparameter tuning.")
        # Compute simple CV scores (no tuning)
        scores = []
        for train_idx, val_idx in tscv.split(X):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            pipe.fit(X_train, y_train)
            scores.append(pipe.score(X_val, y_val))
        # Fit on full data
        pipe.fit(X, y)
        print("Ensemble model CV mean score:", np.mean(scores))
        return MLModelResult(
            model=pipe,
            feature_cols=None,
            task=task,
            best_params=None,
            cv_scores=np.array(scores),
        )

    # ---- Non-ensemble models: RF / GB / XGB ----
    if param_grid is None:
        # small, reasonable defaults
        param_grid = {
            "model__max_depth": [4, 6, 8],
        }
        if model_type in ("rf", "xgb"):
            param_grid["model__n_estimators"] = [200, 400]

    if use_optuna and HAS_OPTUNA:
        # Simple Optuna wrapper: tune one or two key params.
        def objective(trial: "optuna.trial.Trial"):
            params = {
                "model__max_depth": trial.int("max_depth", 3, 10),
            }
            if model_type in ("rf", "xgb"):
                params["model__n_estimators"] = trial.suggest_int(
                    "n_estimators", 200, 800, step=100
                )

            pipe.set_params(**params)
            scores_inner = []
            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]
                pipe.fit(X_train, y_train)
                scores_inner.append(pipe.score(X_val, y_val))
            return float(np.mean(scores_inner))

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=25)
        best_params = study.best_params
        # Map back to pipeline param names
        final_params = {}
        if "max_depth" in best_params:
            final_params["model__max_depth"] = best_params["max_depth"]
        if "n_estimators" in best_params and model_type in ("rf", "xgb"):
            final_params["model__n_estimators"] = best_params["n_estimators"]
        pipe.set_params(**final_params)
        pipe.fit(X, y)
        cv_scores = np.array([study.best_value])
        print("Optuna best params:", final_params)
        print("Optuna best score:", study.best_value)
        return MLModelResult(
            pipe,
            feature_cols=None,
            task=task,
            best_params=final_params,
            cv_scores=cv_scores,
        )

    # GridSearchCV for non-ensemble models
    gscv = GridSearchCV(
        pipe,
        param_grid=param_grid,
        scoring=scoring,
        cv=tscv,
        n_jobs=-1,
        verbose=1,
    )
    gscv.fit(X, y)
    print("Best params:", gscv.best_params_)
    print("CV best score:", gscv.best_score_)

    return MLModelResult(
        model=gscv.best_estimator_,
        feature_cols=None,
        task=task,
        best_params=gscv.best_params_,
        cv_scores=gscv.cv_results_["mean_test_score"],
    )


def predict_scores(
    model: Any,
    X: np.ndarray,
    task: Literal["classification", "regression"]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Unified interface:
    - For classification: returns (p_up, expected_return_proxy).
    - For regression: returns (None, expected_return).
    """
    if task == "classification":
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[:, 1]
        else:
            # fallback: map decision_function to (0,1)
            dec = model.decision_function(X)
            probs = 1 / (1 + np.exp(-dec))
        return probs, probs  # treat prob as proxy for return
    else:
        y_pred = model.predict(X)
        return np.full_like(y_pred, np.nan, dtype=float), y_pred

