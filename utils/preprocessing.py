"""
utils/preprocessing.py
Duomenų paruošimas ML modeliams.
"""
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import joblib, os

SCALER_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "scaler.pkl")

FEATURE_COLS = [
    "tenure",
    "monthly_charges",
    "total_charges",
    "senior_citizen",
    "service_count",
    "has_streaming",
    "has_security_services",
    "is_long_term_contract",
    "charges_per_month_ratio",
    "avg_monthly_over_tenure",
    "high_value_customer",
]


def prepare_data(df: pd.DataFrame, test_size: float = 0.2, use_smote: bool = True):
    """
    Paruošia duomenis treniravimui.
    Grąžina: X_train, X_test, y_train, y_test (numpy arrays).
    """
    X = df[FEATURE_COLS].values
    y = df["churn"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # Išsaugome scaler naudotojo prognozėms vėliau
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)
    joblib.dump(scaler, SCALER_PATH)              # scaler objekto paėmimas ir išsaugojimas į scaler.pkl

    if use_smote:
        try:
            sm = SMOTE(random_state=42)
            X_train, y_train = sm.fit_resample(X_train, y_train)
        except Exception:
            pass  # jei per mažai duomenų SMOTE, tęsiame be jo

    return X_train, X_test, y_train, y_test


def scale_single(input_dict: dict) -> np.ndarray:
    """Paruošia vieno kliento duomenis prognozei."""
    scaler = joblib.load(SCALER_PATH)
    row = np.array([[input_dict[f] for f in FEATURE_COLS]], dtype=float)
    return scaler.transform(row)
