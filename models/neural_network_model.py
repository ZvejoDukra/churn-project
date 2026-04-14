"""
models/neural_network_model.py
Feed Forward neuroninis tinklas su 25 hyperparametrų eksperimentais
"""
import os, json, numpy as np, joblib
os.environ["KERAS_BACKEND"] = "torch"

import keras
from keras import layers, regularizers
from sklearn.metrics import roc_auc_score, f1_score
from utils.metrics import compute_metrics, save_model_result, save_hyperparam_experiment

MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "nn_model.keras")


# ---------------------------------------------------------------------------
# 25 hyperparametrų konfigūracijų lentelė
# ---------------------------------------------------------------------------
EXPERIMENTS = [
    # exp_no, layers_str, lr,  batch, optimizer,epochs,dropout,activation
    (1,  "64",            0.001,  32,  "adam",    30,  0.0,  "relu"),
    (2,  "128",           0.001,  32,  "adam",    30,  0.0,  "relu"),
    (3,  "128-64",        0.001,  32,  "adam",    30,  0.0,  "relu"),
    (4,  "128-64",        0.001,  64,  "adam",    30,  0.0,  "relu"),
    (5,  "128-64",        0.001,  32,  "adam",    50,  0.0,  "relu"),
    (6,  "128-64",        0.001,  32,  "adam",    30,  0.2,  "relu"),
    (7,  "128-64",        0.001,  32,  "adam",    30,  0.4,  "relu"),
    (8,  "128-64",        0.0001, 32,  "adam",    50,  0.2,  "relu"),
    (9,  "128-64",        0.01,   32,  "adam",    30,  0.0,  "relu"),
    (10, "256-128-64",    0.001,  32,  "adam",    30,  0.2,  "relu"),
    (11, "256-128-64",    0.001,  64,  "adam",    50,  0.3,  "relu"),
    (12, "256-128-64",    0.001,  32,  "rmsprop", 30,  0.2,  "relu"),
    (13, "256-128-64",    0.001,  32,  "sgd",     50,  0.0,  "relu"),
    (14, "128-64",        0.001,  32,  "adam",    30,  0.2,  "tanh"),
    (15, "128-64",        0.001,  32,  "adam",    30,  0.2,  "elu"),
    (16, "128-64-32",     0.001,  32,  "adam",    30,  0.2,  "relu"),
    (17, "128-64-32",     0.001,  16,  "adam",    30,  0.2,  "relu"),
    (18, "128-64-32",     0.001,  128, "adam",    30,  0.2,  "relu"),
    (19, "512-256-128",   0.0001, 32,  "adam",    50,  0.3,  "relu"),
    (20, "512-256-128",   0.001,  64,  "adam",    50,  0.4,  "relu"),
    (21, "128-64-32",     0.001,  32,  "adamax",  30,  0.2,  "relu"),
    (22, "128-64-32",     0.001,  32,  "nadam",   30,  0.2,  "relu"),
    (23, "256-128-64-32", 0.001,  32,  "adam",    50,  0.3,  "relu"),
    (24, "256-128-64-32", 0.0005, 64,  "adam",    60,  0.3,  "relu"),
    (25, "256-128-64-32", 0.001,  32,  "adam",    60,  0.2,  "elu"),
]


def _build_model(input_dim: int, layers_str: str, lr: float, optimizer_name: str,
                 dropout: float, activation: str) -> keras.Model:
    """Sukuria  modelį pagal parametrus."""
    units = [int(u) for u in layers_str.split("-")]

    model = keras.Sequential()
    model.add(keras.Input(shape=(input_dim,)))

    for u in units:
        model.add(layers.Dense(u, activation=activation,
                               kernel_regularizer=regularizers.l2(1e-4)))
        if dropout > 0:
            model.add(layers.Dropout(dropout))

    model.add(layers.Dense(1, activation="sigmoid"))

    optimizers = {
        "adam":    keras.optimizers.Adam(lr),
        "rmsprop": keras.optimizers.RMSprop(lr),
        "sgd":     keras.optimizers.SGD(lr, momentum=0.9),
        "adamax":  keras.optimizers.Adamax(lr),
        "nadam":   keras.optimizers.Nadam(lr),
    }
    opt = optimizers.get(optimizer_name, keras.optimizers.Adam(lr))
    model.compile(optimizer=opt, loss="binary_crossentropy",
                  metrics=["accuracy"])
    return model


class NeuralNetworkModel:
    """Feed Forward NN klientų išėjimo prognozavimui"""

    def __init__(self):
        self.model: keras.Model | None = None
        self.metrics: dict = {}
        self.history = None

    def train_best(self, X_train, y_train, X_test, y_test,
                   progress_callback=None) -> dict:
        """
        Apmoko GERIAUSIĄ konfigūraciją (exp 24):
        256-128-64-32, adam, lr=0.0005, batch=64, dropout=0.3, epochs=60
        Naudojama galutiniam modeliui
        """
        input_dim = X_train.shape[1]
        self.model = _build_model(input_dim, "256-128-64-32",
                                   0.0005, "adam", 0.3, "relu")
        cb = [keras.callbacks.EarlyStopping(patience=8, restore_best_weights=True)]
        self.history = self.model.fit(
            X_train, y_train,
            validation_data=(X_test, y_test),
            epochs=60, batch_size=64,
            callbacks=cb, verbose=0
        )
        self.metrics = self._eval(X_test, y_test)
        save_model_result("NeuralNetwork", self.metrics,
                          notes={"layers": "256-128-64-32", "lr": 0.0005,
                                 "batch_size": 64, "optimizer": "adam",
                                 "dropout": 0.3, "epochs": 60})
        return self.metrics

    def run_all_experiments(self, X_train, y_train, X_test, y_test,
                            progress_callback=None) -> list[dict]:
        """
        Paleidžia visus 25 eksperimentus
        progress_callback(exp_no, total) → naudojamas Streamlit progress bar
        Grąžina sąrašą su rezultatais
        """
        results = []
        input_dim = X_train.shape[1]

        for (exp_no, layers_str, lr, batch, opt_name,
             epochs, dropout, activation) in EXPERIMENTS:

            m = _build_model(input_dim, layers_str, lr, opt_name,
                             dropout, activation)
            cb = [keras.callbacks.EarlyStopping(patience=5,
                                                restore_best_weights=True)]
            hist = m.fit(X_train, y_train,
                         validation_data=(X_test, y_test),
                         epochs=epochs, batch_size=batch,
                         callbacks=cb, verbose=0)

            y_pred = (m.predict(X_test, verbose=0) > 0.5).astype(int).flatten()
            y_prob = m.predict(X_test, verbose=0).flatten()
            metrics = compute_metrics(y_test, y_pred, y_prob)
            metrics["val_loss"] = float(hist.history["val_loss"][-1])

            params = {
                "layers": layers_str, "neurons": layers_str,
                "learning_rate": lr, "batch_size": batch,
                "optimizer": opt_name, "epochs": epochs,
                "dropout": dropout, "activation": activation
            }
            save_hyperparam_experiment(exp_no, params, metrics)

            results.append({"exp_no": exp_no, **params, **metrics})

            if progress_callback:
                progress_callback(exp_no, len(EXPERIMENTS))

        return results

    def _eval(self, X_test, y_test) -> dict:
        y_prob = self.model.predict(X_test, verbose=0).flatten()
        y_pred = (y_prob > 0.5).astype(int)
        return compute_metrics(y_test, y_pred, y_prob)

    def predict(self, X) -> tuple[int, float]:
        prob = float(self.model.predict(X, verbose=0)[0, 0])
        return int(prob > 0.5), prob

    def save(self) -> None:
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        self.model.save(MODEL_PATH)

    def load(self) -> bool:
        if os.path.exists(MODEL_PATH):
            self.model = keras.models.load_model(MODEL_PATH)
            return True
        return False

    @staticmethod
    def is_trained() -> bool:
        return os.path.exists(MODEL_PATH)
