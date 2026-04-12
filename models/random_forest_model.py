"""
models/random_forest_model.py
Random Forest modelis - pagrindinis ML modelis.
"""
import joblib, os, numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from utils.metrics import compute_metrics, save_model_result

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "rf_model.pkl")


class RandomForestModel:
    """Random Forest klasifikatorius klientų išėjimo prognozavimui."""

    def __init__(self, n_estimators: int = 200, max_depth: int = 10,
                 min_samples_split: int = 5, random_state: int = 42):
        self.model = RandomForestClassifier(
            n_estimators      = n_estimators,
            max_depth         = max_depth,
            min_samples_split = min_samples_split,
            class_weight      = "balanced",
            random_state      = random_state,
            n_jobs            = -1,
        )
        self.metrics: dict = {}
        self.feature_importances_: np.ndarray | None = None

    def train(self, X_train, y_train) -> None:
        """Apmoko modelį."""
        self.model.fit(X_train, y_train)
        self.feature_importances_ = self.model.feature_importances_

    def evaluate(self, X_test, y_test) -> dict:
        """Įvertina modelį ir išsaugo metrikas."""
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        self.metrics = compute_metrics(y_test, y_pred, y_prob)
        save_model_result("RandomForest", self.metrics,
                          notes={"n_estimators": self.model.n_estimators,
                                 "max_depth": self.model.max_depth})
        return self.metrics

    def predict(self, X) -> tuple[int, float]:
        """Grąžina (klasė, tikimybė)."""
        cls  = int(self.model.predict(X)[0])
        prob = float(self.model.predict_proba(X)[0, 1])
        return cls, prob

    def save(self) -> None:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        joblib.dump(self.model, MODEL_PATH)

    def load(self) -> bool:
        if os.path.exists(MODEL_PATH):
            self.model = joblib.load(MODEL_PATH)
            return True
        return False

    @staticmethod
    def is_trained() -> bool:
        return os.path.exists(MODEL_PATH)
