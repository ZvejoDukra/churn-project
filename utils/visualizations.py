"""
utils/visualizations.py
Visi grafikai — 7 prasmingos vizualizacijos.
SQLAlchemy 2.0: select() + session.execute() — be session.query().
"""
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sqlalchemy import select, func
from database.connection import get_session
from database.models import Customer, HyperparamExperiment


def _fetch_all_customers() -> list:
    """Grąžina visus klientus iš DB. SQLAlchemy 2.0 sintaksė."""
    session = get_session()
    stmt = select(Customer)
    customers = session.execute(stmt).scalars().all()
    session.close()
    return customers


def plot_churn_distribution() -> go.Figure:
    """Stulpelinė diagrama: klientų išėjimo pasiskirstymas."""
    customers = _fetch_all_customers()
    churn_counts = {"Liko (0)": 0, "Išėjo (1)": 0}
    for c in customers:
        if c.churn == 1:
            churn_counts["Išėjo (1)"] += 1
        else:
            churn_counts["Liko (0)"] += 1

    total = sum(churn_counts.values())
    fig = go.Figure(go.Bar(
        x=list(churn_counts.keys()),
        y=list(churn_counts.values()),
        marker_color=["#2196F3", "#F44336"],
        text=[f"{v} ({v/total*100:.1f}%)" for v in churn_counts.values()],
        textposition="outside",
    ))
    fig.update_layout(
        title="Klientų išėjimo pasiskirstymas",
        yaxis_title="Klientų skaičius",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
    )
    return fig


def plot_charges_by_contract() -> go.Figure:
    """Box plot: mėnesinės išlaidos pagal churn ir sutarties tipą."""
    customers = _fetch_all_customers()
    df = pd.DataFrame({
        "contract":        [c.contract for c in customers],
        "monthly_charges": [c.monthly_charges for c in customers],
        "churn":           ["Išėjo" if c.churn == 1 else "Liko" for c in customers],
    })
    fig = px.box(df, x="contract", y="monthly_charges", color="churn",
                 title="Mėnesinės išlaidos pagal sutarties tipą ir churn",
                 labels={"contract": "Sutarties tipas",
                         "monthly_charges": "Mėnesinės išlaidos (€)",
                         "churn": "Statusas"},
                 color_discrete_map={"Liko": "#2196F3", "Išėjo": "#F44336"})
    fig.update_layout(plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def plot_tenure_churn() -> go.Figure:
    """Histograma: stažo pasiskirstymas pagal churn grupę."""
    customers = _fetch_all_customers()
    stayed  = [c.tenure for c in customers if c.churn == 0]
    churned = [c.tenure for c in customers if c.churn == 1]

    fig = go.Figure()
    fig.add_trace(go.Histogram(x=stayed,  name="Liko",  marker_color="#2196F3",
                               opacity=0.7, nbinsx=30))
    fig.add_trace(go.Histogram(x=churned, name="Išėjo", marker_color="#F44336",
                               opacity=0.7, nbinsx=30))
    fig.update_layout(
        barmode="overlay",
        title="Kliento stažo pasiskirstymas pagal churn",
        xaxis_title="Stažas (mėnesiai)",
        yaxis_title="Klientų skaičius",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_services_vs_churn() -> go.Figure:
    """Dviguba ašis: paslaugų skaičiaus įtaka churn."""
    customers = _fetch_all_customers()
    df = pd.DataFrame({
        "service_count": [c.service_count for c in customers],
        "churn":         [c.churn for c in customers],
    })
    grouped = df.groupby("service_count")["churn"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["service_count", "churn_rate", "count"]

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=grouped["service_count"], y=grouped["count"],
                         name="Klientų sk.", marker_color="#90CAF9", opacity=0.6),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=grouped["service_count"],
                             y=grouped["churn_rate"] * 100,
                             name="Churn %", mode="lines+markers",
                             marker=dict(color="#F44336", size=8),
                             line=dict(color="#F44336", width=2)),
                  secondary_y=True)
    fig.update_xaxes(title_text="Paslaugų skaičius")
    fig.update_yaxes(title_text="Klientų skaičius", secondary_y=False)
    fig.update_yaxes(title_text="Churn procentas (%)", secondary_y=True)
    fig.update_layout(title="Paslaugų skaičiaus įtaka klientų išėjimui",
                      plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    return fig


def plot_confusion_matrix(cm: list, model_name: str) -> go.Figure:
    """Heatmap: confusion matrix."""
    cm_arr = np.array(cm)
    labels = ["Liko (0)", "Išėjo (1)"]
    fig = go.Figure(go.Heatmap(
        z=cm_arr, x=labels, y=labels,
        colorscale="Blues", text=cm_arr,
        texttemplate="%{text}", showscale=True,
    ))
    fig.update_layout(
        title=f"Confusion Matrix — {model_name}",
        xaxis_title="Prognozuota",
        yaxis_title="Tikroji reikšmė",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_experiment_results() -> go.Figure:
    """Interaktyvi lentelė: visi 25 neuroninio tinklo eksperimentai."""
    session = get_session()
    stmt = select(HyperparamExperiment).order_by(HyperparamExperiment.experiment_no)
    exps = session.execute(stmt).scalars().all()
    session.close()

    if not exps:
        return go.Figure().update_layout(title="Eksperimentų duomenų nėra")

    rows = [{
        "Exp#":      e.experiment_no,
        "Sluoksniai": e.layers,
        "LR":         e.learning_rate,
        "Batch":      e.batch_size,
        "Optimizer":  e.optimizer,
        "Epochs":     e.epochs,
        "Dropout":    e.dropout,
        "Aktiv.":     e.activation,
        "Accuracy":   f"{(e.val_accuracy or 0):.4f}",
        "F1":         f"{(e.val_f1 or 0):.4f}",
        "ROC-AUC":    f"{(e.val_roc_auc or 0):.4f}",
    } for e in exps]

    df = pd.DataFrame(rows)
    fig = go.Figure(go.Table(
        header=dict(values=list(df.columns),
                    fill_color="#1565C0",
                    font=dict(color="white", size=11),
                    align="center"),
        cells=dict(values=[df[c] for c in df.columns],
                   fill_color=[["#E3F2FD" if i % 2 == 0 else "white"
                                for i in range(len(df))]],
                   align="center", font_size=11)
    ))
    fig.update_layout(title="Visų 25 neuroninio tinklo eksperimentų rezultatai",
                      paper_bgcolor="rgba(0,0,0,0)")
    return fig


def plot_feature_importance(importances: np.ndarray, feature_names: list) -> go.Figure:
    """Stulpelinė diagrama: Random Forest feature importance."""
    idx = np.argsort(importances)[::-1]
    fig = go.Figure(go.Bar(
        x=[feature_names[i] for i in idx],
        y=[importances[i] for i in idx],
        marker_color="#1976D2",
    ))
    fig.update_layout(
        title="Feature importance (Random Forest)",
        xaxis_title="Požymis",
        yaxis_title="Svarba",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig
