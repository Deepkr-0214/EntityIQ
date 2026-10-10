"""
Model training module.

Supports Logistic Regression baseline and LightGBM for stronger performance.
"""
import numpy as np
import pandas as pd
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
from src.config import MODELS_DIR, RANDOM_SEED


def train_logistic_regression(X_train, y_train, feature_names=None):
    """Train a logistic regression baseline model."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)

    model = LogisticRegression(
        max_iter=1000,
        random_state=RANDOM_SEED,
        class_weight='balanced',
        C=1.0,
    )
    model.fit(X_scaled, y_train)

    if feature_names is not None:
        importances = sorted(zip(feature_names, model.coef_[0]),
                             key=lambda x: abs(x[1]), reverse=True)
        print("\nTop feature coefficients (Logistic Regression):")
        for name, coef in importances[:15]:
            print(f"  {name:30s} {coef:+.4f}")

    return model, scaler


def train_lightgbm(X_train, y_train, X_val=None, y_val=None, feature_names=None):
    """Train a LightGBM model."""
    params = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'verbosity': -1,
        'seed': RANDOM_SEED,
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'is_unbalance': True,
    }

    callbacks = [lgb.log_evaluation(100)]
    if X_val is not None and y_val is not None:
        callbacks.append(lgb.early_stopping(50))

    model = lgb.LGBMClassifier(**params)

    eval_set = [(X_val, y_val)] if X_val is not None else None
    model.fit(
        X_train, y_train,
        eval_set=eval_set,
        callbacks=callbacks,
    )

    if feature_names is not None:
        importances = sorted(zip(feature_names, model.feature_importances_),
                             key=lambda x: x[1], reverse=True)
        print("\nTop feature importances (LightGBM):")
        for name, imp in importances[:15]:
            print(f"  {name:30s} {imp:8d}")

    return model


def predict_proba(model, X, scaler=None):
    """Get match probabilities from a model."""
    if scaler is not None:
        X = scaler.transform(X)
    return model.predict_proba(X)[:, 1]


def save_model(model, name, scaler=None):
    """Save model (and optional scaler) to disk."""
    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump({'model': model, 'scaler': scaler}, f)
    print(f"Model saved to {model_path}")


def load_model(name):
    """Load model (and optional scaler) from disk."""
    model_path = os.path.join(MODELS_DIR, f"{name}.pkl")
    with open(model_path, 'rb') as f:
        data = pickle.load(f)
    return data['model'], data.get('scaler')
