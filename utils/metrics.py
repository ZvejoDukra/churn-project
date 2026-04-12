"""
utils/metrics.py
Visų metrikų skaičiavimas ir išsaugojimas DB.
SQLAlchemy 2.0: session.add() + session.commit() — be session.query().
"""
import json, uuid
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    classification_report, matthews_corrcoef
)
from database.connection import get_session
from database.models import ModelResult, HyperparamExperiment


def compute_metrics(y_true, y_pred, y_prob=None) -> dict:
    """Apskaičiuoja visas metrikas ir grąžina kaip žodyną."""
    metrics = {
        "accuracy":         round(accuracy_score(y_true, y_pred), 4),
        "precision":        round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall":           round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1":               round(f1_score(y_true, y_pred, zero_division=0), 4),
        "mcc":              round(matthews_corrcoef(y_true, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
        "report":           classification_report(y_true, y_pred, output_dict=True),
    }
    metrics["roc_auc"] = round(roc_auc_score(y_true, y_prob), 4) if y_prob is not None else None
    return metrics


def save_model_result(model_name: str, metrics: dict, notes: dict = None) -> None:
    """
    Išsaugo modelio rezultatus DB.
    SQLAlchemy 2.0: session.add() + session.commit().
    """
    session = get_session()
    result = ModelResult(
        model_name      = model_name,
        experiment_id   = str(uuid.uuid4())[:8],
        accuracy        = metrics.get("accuracy"),
        precision_score = metrics.get("precision"),
        recall          = metrics.get("recall"),
        f1              = metrics.get("f1"),
        roc_auc         = metrics.get("roc_auc"),
        notes           = json.dumps(notes or {}, ensure_ascii=False),
    )
    session.add(result)
    session.commit()
    session.close()


def save_hyperparam_experiment(exp_no: int, params: dict, metrics: dict) -> None:
    """
    Išsaugo vieną neuroninio tinklo eksperimentą DB.
    SQLAlchemy 2.0: session.add() + session.commit().
    """
    session = get_session()
    exp = HyperparamExperiment(
        experiment_no = exp_no,
        layers        = params.get("layers", ""),
        neurons       = params.get("neurons", ""),
        learning_rate = params.get("learning_rate"),
        batch_size    = params.get("batch_size"),
        optimizer     = params.get("optimizer", ""),
        epochs        = params.get("epochs"),
        dropout       = params.get("dropout"),
        activation    = params.get("activation", ""),
        val_accuracy  = metrics.get("accuracy"),
        val_loss      = metrics.get("val_loss"),
        val_f1        = metrics.get("f1"),
        val_roc_auc   = metrics.get("roc_auc"),
        notes         = json.dumps(params, ensure_ascii=False),
    )
    session.add(exp)
    session.commit()
    session.close()
