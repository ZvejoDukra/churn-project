"""
app.py
Pagrindinis Streamlit frontentas — daugelio puslapių programa.
SQLAlchemy 2.0: select() + session.execute() — be session.query().
Paleidimas: streamlit run app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import json, os, sys

sys.path.insert(0, os.path.dirname(__file__))

from sqlalchemy import select, func
from database.connection import init_db, db_exists, get_session
from database.models import Customer, ModelResult, Prediction
from utils.data_loader import load_csv_to_db, load_data_from_db
from utils.preprocessing import prepare_data, scale_single, FEATURE_COLS
from utils.visualizations import (
    plot_churn_distribution, plot_charges_by_contract,
    plot_tenure_churn, plot_services_vs_churn,
    plot_confusion_matrix, plot_experiment_results,
    plot_feature_importance,
)
from models.random_forest_model import RandomForestModel
from models.neural_network_model import NeuralNetworkModel

st.set_page_config(
    page_title="Churn Prognozavimas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

init_db()


# ---------------------------------------------------------------------------
# DB skaičiavimo pagalbinės funkcijos (SQLAlchemy 2.0)
# ---------------------------------------------------------------------------

def count_table(model_class) -> int:
    """Grąžina eilučių skaičių iš nurodytos lentelės. SQLAlchemy 2.0."""
    session = get_session()
    stmt = select(func.count()).select_from(model_class)
    count = session.execute(stmt).scalar()
    session.close()
    return count or 0


def fetch_customers_preview(limit: int = 10) -> list:
    """Grąžina pirmus N klientų iš DB. SQLAlchemy 2.0."""
    session = get_session()
    stmt = select(Customer).limit(limit)
    customers = session.execute(stmt).scalars().all()
    session.close()
    return customers


def fetch_latest_model_result(model_name: str) -> ModelResult | None:
    """Grąžina naujausią modelio rezultatą pagal pavadinimą. SQLAlchemy 2.0."""
    session = get_session()
    stmt = (
        select(ModelResult)
        .where(ModelResult.model_name == model_name)
        .order_by(ModelResult.trained_at.desc())
        .limit(1)
    )
    result = session.execute(stmt).scalars().first()
    session.close()
    return result


def fetch_all_model_results() -> list:
    """Grąžina visus modelių rezultatus. SQLAlchemy 2.0."""
    session = get_session()
    stmt = select(ModelResult)
    results = session.execute(stmt).scalars().all()
    session.close()
    return results


def save_prediction(model_name: str, input_data: dict,
                    probability: float, prediction: int) -> None:
    
    try:
        session = get_session()
        pred = Prediction(
            model_used        = model_name,
            input_data        = json.dumps(input_data),
            churn_probability = probability,
            prediction        = prediction,
        )
        session.add(pred)
        session.commit()
        session.close()
    except Exception as e:
        print(f"Klaida išsaugant prognozę: {e}")


# ---------------------------------------------------------------------------
# Šoninė juosta
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("📊 Churn Prognozavimas")
    st.markdown("---")
    page = st.radio(
        "Navigacija",
        [
            "🏠 Pradžia",
            "📥 Duomenų įkėlimas",
            "🤖 Modelių treniravimas",
            "🔮 Prognozavimas",
            "📈 Grafikai",
            "🧪 Eksperimentai",
            "📋 Išvados",
        ],
        label_visibility="collapsed",
    )
    st.markdown("---")
    db_ok = db_exists()
    st.markdown(f"**DB būsena:** {'✅ Duomenys įkelti' if db_ok else '⚠️ Tuščia DB'}")
    rf_ok = RandomForestModel.is_trained()
    nn_ok = NeuralNetworkModel.is_trained()
    st.markdown(f"**Random Forest:** {'✅ Apmokytas' if rf_ok else '❌ Neapmokytas'}")
    st.markdown(f"**Neuroninis tinklas:** {'✅ Apmokytas' if nn_ok else '❌ Neapmokytas'}")


# ===========================================================================
# 1. PRADŽIA
# ===========================================================================
if page == "🏠 Pradžia":
    st.title("🏠 Klientų išėjimo prognozavimo sistema")
    st.markdown("""
    Ši sistema leidžia jums:
    - 📥 **Įkelti** telekomunikacijų klientų CSV duomenis į SQLite duomenų bazę
    - 🤖 **Apmokyti** Random Forest ir Feed Forward neuroninį tinklą
    - 🔮 **Gauti prognozes** individualiems klientams
    - 📈 **Peržiūrėti grafikus** ir analizę
    - 🧪 **Palyginti** 25 neuroninio tinklo konfigūracijų rezultatus
    """)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Duomenų bazėje klientų", count_table(Customer))
    with col2:
        st.metric("Atliktų eksperimentų", count_table(ModelResult))
    with col3:
        st.metric("Prognozių skaičius", count_table(Prediction))

    st.info("👈 Naudokite šoninę juostą navigacijai. Pradėkite nuo **Duomenų įkėlimo**.")


# ===========================================================================
# 2. DUOMENŲ ĮKĖLIMAS
# ===========================================================================
elif page == "📥 Duomenų įkėlimas":
    st.title("📥 Duomenų įkėlimas")
    st.markdown("""
    Įkelkite **Telco Customer Churn** CSV failą iš Kaggle arba bet kurį suderinamą CSV.
    Reikalingi stulpeliai: `customerID`, `tenure`, `MonthlyCharges`, `TotalCharges`, `Churn`.
    """)

    uploaded = st.file_uploader("Pasirinkite CSV failą", type=["csv"])

    if uploaded:
        tmp_path = os.path.join("data", "uploaded_temp.csv")
        os.makedirs("data", exist_ok=True)
        with open(tmp_path, "wb") as f:
            f.write(uploaded.read())

        st.markdown("### Duomenų peržiūra (pirmos 5 eilutės)")
        df_preview = pd.read_csv(tmp_path)
        st.dataframe(df_preview.head(), use_container_width=True)
        st.markdown(f"**Eilučių:** {len(df_preview)} | **Stulpelių:** {len(df_preview.columns)}")

        if st.button("⬆️ Įkelti į duomenų bazę", type="primary"):
            with st.spinner("Vykdome transformacijas ir įkeliame į SQLite..."):
                success, msg = load_csv_to_db(tmp_path)
            if success:
                st.success(f"✅ {msg}")
                st.balloons()
            else:
                st.error(f"❌ {msg}")

    st.markdown("---")
    st.markdown("### 🗄️ Dabartiniai duomenys DB")
    if db_exists():
        customers = fetch_customers_preview(10)
        df_db = pd.DataFrame([{
            "ID":            c.customer_id,
            "Stažas":        c.tenure,
            "Mėn. išlaidos": c.monthly_charges,
            "Paslaugų sk.":  c.service_count,
            "Ilga sutartis": c.is_long_term_contract,
            "High-value":    c.high_value_customer,
            "Churn":         c.churn,
        } for c in customers])
        st.dataframe(df_db, use_container_width=True)
    else:
        st.warning("Duomenų bazė tuščia. Įkelkite CSV failą.")


# ===========================================================================
# 3. MODELIŲ TRENIRAVIMAS
# ===========================================================================
elif page == "🤖 Modelių treniravimas":
    st.title("🤖 Modelių treniravimas")

    if not db_exists():
        st.error("❌ Pirmiausia įkelkite duomenis (Duomenų įkėlimas).")
        st.stop()

    tab1, tab2 = st.tabs(["🌲 Random Forest", "🧠 Neuroninis tinklas"])

    with tab1:
        st.markdown("### Random Forest parametrai")
        col1, col2 = st.columns(2)
        with col1:
            n_est = st.slider("Medžių skaičius (n_estimators)", 50, 500, 200, 50)
            max_d = st.slider("Gylis (max_depth)", 3, 20, 10)
        with col2:
            min_s = st.slider("Min. skaidymui (min_samples_split)", 2, 20, 5)

        if st.button("🚀 Trenuoti Random Forest", type="primary"):
            with st.spinner("Apmokoma..."):
                df = load_data_from_db()
                X_train, X_test, y_train, y_test = prepare_data(df)
                rf = RandomForestModel(n_estimators=n_est, max_depth=max_d,
                                      min_samples_split=min_s)
                rf.train(X_train, y_train)
                metrics = rf.evaluate(X_test, y_test)
                rf.save()

            st.success("✅ Random Forest apmokytas ir išsaugotas!")
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Accuracy",  f"{metrics['accuracy']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col3.metric("Recall",    f"{metrics['recall']:.4f}")
            col4.metric("F1",        f"{metrics['f1']:.4f}")
            if metrics.get("roc_auc"):
                st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
            st.metric("MCC", f"{metrics.get('mcc', 0):.4f}")
            if metrics.get("oob_score"):                                # įterpiama papildoma eilutė
                st.metric("OOB Score", f"{metrics['oob_score']:.4f}")   # įtrepiama papildoma eilutė
            st.plotly_chart(plot_confusion_matrix(
                metrics["confusion_matrix"], "Random Forest"),
                use_container_width=True)

            if rf.feature_importances_ is not None:
                st.plotly_chart(plot_feature_importance(
                    rf.feature_importances_, FEATURE_COLS),
                    use_container_width=True)

    with tab2:
        st.markdown("### Neuroninio tinklo treniravimas")
        mode = st.radio("Režimas",
                        ["Tik geriausias modelis (greita)",
                         "Visi 25 eksperimentai (lėta ~10–20 min)"])

        if st.button("🚀 Trenuoti neuroninį tinklą", type="primary"):
            df = load_data_from_db()
            X_train, X_test, y_train, y_test = prepare_data(df)
            nn = NeuralNetworkModel()

            if "greita" in mode:
                with st.spinner("Apmokoma geriausia konfigūracija..."):
                    metrics = nn.train_best(X_train, y_train, X_test, y_test)
                nn.save()
                st.success("✅ Neuroninis tinklas apmokytas!")
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Accuracy",  f"{metrics['accuracy']:.4f}")
                col2.metric("Precision", f"{metrics['precision']:.4f}")
                col3.metric("Recall",    f"{metrics['recall']:.4f}")
                col4.metric("F1",        f"{metrics['f1']:.4f}")
                if metrics.get("roc_auc"):
                    st.metric("ROC-AUC", f"{metrics['roc_auc']:.4f}")
                st.plotly_chart(plot_confusion_matrix(
                    metrics["confusion_matrix"], "Neuroninis tinklas"),
                    use_container_width=True)
            else:
                progress = st.progress(0)
                status   = st.empty()

                def cb(exp_no, total):
                    progress.progress(exp_no / total)
                    status.markdown(f"Eksperimentas **{exp_no}/{total}**...")

                with st.spinner("Vykdomi 25 eksperimentai..."):
                    results = nn.run_all_experiments(
                        X_train, y_train, X_test, y_test, progress_callback=cb)
                    nn.train_best(X_train, y_train, X_test, y_test)
                    nn.save()

                progress.progress(1.0)
                status.success("✅ Visi 25 eksperimentai atlikti!")
                df_res = pd.DataFrame(results)
                st.dataframe(df_res[[
                    "exp_no", "layers", "learning_rate", "batch_size",
                    "optimizer", "epochs", "dropout", "activation",
                    "accuracy", "f1", "roc_auc"
                ]].round(4), use_container_width=True)


# ===========================================================================
# 4. PROGNOZAVIMAS
# ===========================================================================

elif page == "🔮 Prognozavimas":
    st.title("🔮 Kliento išėjimo prognozavimas")
    tab_single, tab_batch = st.tabs(["👤 Vienas klientas", "📋 Batch (CSV)"])

    with tab_batch:
        st.markdown("### CSV failo įkėlimas")
        batch_file = st.file_uploader("Įkelkite CSV su klientais", type=["csv"], key="batch")
        if batch_file:
            df_batch = pd.read_csv(batch_file)
            st.dataframe(df_batch.head(), use_container_width=True)
            model_batch = st.selectbox("Modelis", ["Random Forest", "Neuroninis tinklas"], key="batch_model")
            if st.button("🔮 Prognozuoti visus", type="primary"):
                from utils.preprocessing import scale_single, FEATURE_COLS
                results = []
                for _, row in df_batch.iterrows():
                    try:
                        input_dict = {
                            "tenure":                  float(row.get("tenure", 0)),
                            "monthly_charges":         float(row.get("MonthlyCharges", 0)),
                            "total_charges":           float(row.get("TotalCharges", 0)),
                            "senior_citizen":          int(row.get("SeniorCitizen", 0)),
                            "service_count":           int(row.get("service_count", 3)),
                            "has_streaming":           int(row.get("has_streaming", 0)),
                            "has_security_services":   int(row.get("has_security_services", 0)),
                            "is_long_term_contract":   int(row.get("is_long_term_contract", 0)),
                            "charges_per_month_ratio": float(row.get("charges_per_month_ratio", 0)),
                            "avg_monthly_over_tenure": float(row.get("avg_monthly_over_tenure", 0)),
                            "high_value_customer":     int(row.get("high_value_customer", 0)),
                        }
                        X = scale_single(input_dict)
                        if model_batch == "Random Forest":
                            rf = RandomForestModel(); rf.load()
                            cls, prob = rf.predict(X)
                        else:
                            nn = NeuralNetworkModel(); nn.load()
                            cls, prob = nn.predict(X)
                        results.append({
                            "customerID": row.get("customerID", "?"),
                            "Prognozė":   "⚠️ Išeis" if cls == 1 else "✅ Liks",
                            "Tikimybė":   f"{prob:.1%}",
                        })
                        save_prediction(model_batch, input_dict, prob, cls)
                    except Exception as e:
                        results.append({
                            "customerID": row.get("customerID", "?"),
                            "Prognozė": "Klaida",
                            "Tikimybė": str(e)
                        })
                st.dataframe(pd.DataFrame(results), use_container_width=True)

    with tab_single:
        if not (RandomForestModel.is_trained() or NeuralNetworkModel.is_trained()):
            st.error("❌ Pirmiausia aptreniruokite modelius.")
            st.stop()

        model_choice = st.selectbox(
            "Modelio pasirinkimas",
            ["Random Forest", "Neuroninis tinklas"] if (
                RandomForestModel.is_trained() and NeuralNetworkModel.is_trained())
            else (["Random Forest"] if RandomForestModel.is_trained()
                  else ["Neuroninis tinklas"])
        )

        threshold = st.slider(
            "Sprendimo slenkstis",
            min_value=0.3,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="0.5 = default. Didesnė reikšmė = aukštesni reikalavimai kad klientas būtų laikomas 'išeinančiu'"
        )

        st.markdown("### Kliento duomenys")
        col1, col2, col3 = st.columns(3)
        with col1:
            tenure = st.slider("Stažas (mėnesiai)", 0, 72, 12)
            monthly_charges = st.number_input("Mėnesinės išlaidos (€)", 0.0, 200.0, 65.0)
            total_charges = st.number_input(
                "Iš viso sumokėta (€)",
                0.0,
                10000.0,
                float(monthly_charges * tenure)
            )
            senior_citizen = st.selectbox(
                "Vyresnio amžiaus?", [0, 1],
                format_func=lambda x: "Taip" if x else "Ne"
            )

        with col2:
            service_count = st.slider("Paslaugų skaičius", 0, 8, 3)
            has_streaming = st.selectbox(
                "Turi srautinį turinį?", [0, 1],
                format_func=lambda x: "Taip" if x else "Ne"
            )
            has_security_services = st.selectbox(
                "Turi saugumo paslaugas?", [0, 1],
                format_func=lambda x: "Taip" if x else "Ne"
            )

        with col3:
            is_long_term_contract = st.selectbox(
                "Ilgalaikė sutartis?", [0, 1],
                format_func=lambda x: "Taip" if x else "Ne"
            )
            high_value_customer = int(monthly_charges > 70)
            charges_per_month_ratio = round(total_charges / (tenure + 1), 4)
            avg_monthly_over_tenure = round(monthly_charges / (tenure + 1), 4)
            st.metric("Vid. mėn. išlaidos / stažas", f"{avg_monthly_over_tenure:.2f}")
            st.metric("High-value klientas", "Taip" if high_value_customer else "Ne")

        if st.button("🔮 Gauti prognozę", type="primary"):
            from utils.preprocessing import scale_single

            input_dict = {
                "tenure":                  tenure,
                "monthly_charges":         monthly_charges,
                "total_charges":           total_charges,
                "senior_citizen":          senior_citizen,
                "service_count":           service_count,
                "has_streaming":           has_streaming,
                "has_security_services":   has_security_services,
                "is_long_term_contract":   is_long_term_contract,
                "charges_per_month_ratio": charges_per_month_ratio,
                "avg_monthly_over_tenure": avg_monthly_over_tenure,
                "high_value_customer":     high_value_customer,
            }

            X = scale_single(input_dict)

            if model_choice == "Random Forest":
                rf = RandomForestModel()
                rf.load()
                cls, prob = rf.predict(X, threshold=threshold)
            else:
                nn = NeuralNetworkModel()
                nn.load()
                cls, prob = nn.predict(X)

            save_prediction(model_choice, input_dict, prob, cls)

            st.markdown("---")
            if cls == 1:
                st.error(f"⚠️ **KLIENTAS GREIČIAUSIAI IŠEIS** — tikimybė: **{prob:.1%}**")
            else:
                st.success(f"✅ **KLIENTAS GREIČIAUSIAI LIKS** — išėjimo tikimybė: **{prob:.1%}**")
            st.progress(prob)
            st.caption(f"Išėjimo tikimybė: {prob:.4f}")




# ===========================================================================
# 5. GRAFIKAI
# ===========================================================================
elif page == "📈 Grafikai":
    st.title("📈 Duomenų vizualizacijos")

    if not db_exists():
        st.error("❌ Nėra duomenų. Pirmiausia įkelkite CSV.")
        st.stop()

    tab1, tab2, tab3, tab4 = st.tabs([
        "Churn pasiskirstymas", "Išlaidos & sutartis",
        "Stažo įtaka", "Paslaugų įtaka",
    ])
    with tab1:
        st.plotly_chart(plot_churn_distribution(), use_container_width=True)
    with tab2:
        st.plotly_chart(plot_charges_by_contract(), use_container_width=True)
    with tab3:
        st.plotly_chart(plot_tenure_churn(), use_container_width=True)
    with tab4:
        st.plotly_chart(plot_services_vs_churn(), use_container_width=True)


# ===========================================================================
# 6. EKSPERIMENTAI
# ===========================================================================
elif page == "🧪 Eksperimentai":
    st.title("🧪 Neuroninio tinklo eksperimentai")
    st.markdown("Čia matote visų 25 hyperparametrų testavimo rezultatus.")
    st.plotly_chart(plot_experiment_results(), use_container_width=True)

    results = fetch_all_model_results()
    if results:
        st.markdown("### Modelių palyginimas")
        df_comp = pd.DataFrame([{
            "Modelis":   r.model_name,
            "Accuracy":  r.accuracy,
            "Precision": r.precision_score,
            "Recall":    r.recall,
            "F1":        r.f1,
            "ROC-AUC":   r.roc_auc,
            "Data":      r.trained_at,
        } for r in results]).round(4)
        st.dataframe(df_comp, use_container_width=True)


# ===========================================================================
# 7. IŠVADOS
# ===========================================================================
elif page == "📋 Išvados":
    st.title("📋 Modelių palyginimas ir išvados")

    rf_results = fetch_latest_model_result("RandomForest")
    nn_results = fetch_latest_model_result("NeuralNetwork")

    if not rf_results or not nn_results:
        st.warning("Reikalingi abu apmokyti modeliai išvadoms.")
        st.stop()

    st.markdown("### 📊 Metrikų palyginimas")
    comp = pd.DataFrame({
        "Metrika":            ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        "Random Forest":      [rf_results.accuracy, rf_results.precision_score,
                               rf_results.recall, rf_results.f1, rf_results.roc_auc],
        "Neuroninis tinklas": [nn_results.accuracy, nn_results.precision_score,
                               nn_results.recall, nn_results.f1, nn_results.roc_auc],
    }).round(4)
    st.dataframe(comp, use_container_width=True, hide_index=True)

    import plotly.graph_objects as go
    metrics_labels = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    rf_vals = [rf_results.accuracy, rf_results.precision_score,
               rf_results.recall, rf_results.f1, rf_results.roc_auc or 0]
    nn_vals = [nn_results.accuracy, nn_results.precision_score,
               nn_results.recall, nn_results.f1, nn_results.roc_auc or 0]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Random Forest",      x=metrics_labels, y=rf_vals,
                         marker_color="#1976D2"))
    fig.add_trace(go.Bar(name="Neuroninis tinklas", x=metrics_labels, y=nn_vals,
                         marker_color="#F44336"))
    fig.update_layout(barmode="group",
                      title="Random Forest vs Neuroninis tinklas",
                      yaxis_title="Reikšmė",
                      plot_bgcolor="rgba(0,0,0,0)",
                      paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📝 Tekstinės išvados")
    winner = "Random Forest" if (rf_results.f1 or 0) >= (nn_results.f1 or 0) \
             else "Neuroninis tinklas"
    st.markdown(f"""
    **Geriau pasirodė: {winner}**

    #### Random Forest
    - Stabilus ir interpretuojamas modelis
    - Feature importance leidžia suprasti, kokie požymiai lemia sprendimą
    - Geras rezultatas su santykinai mažai duomenų

    #### Neuroninis tinklas (Feed Forward)
    - Gali išmokti sudėtingų nelinijinių ryšių
    - Reikalauja daugiau duomenų ir laiko treniravimui
    - Jautresnis hyperparametrų pasirinkimui

    #### Hyperparametrų eksperimentų išvados (25 bandymai)
    - **Sluoksnių skaičius:** 256-128-64-32 dažniausiai geriausias
    - **Learning rate:** 0.0001 - 0.001 optimalus; 0.01 nestabilus
    - **Batch size:** 32 - 64 geriausias balanso taškas
    - **Dropout:** 0.2 - 0.3 sumažino overfittingą
    - **Optimizer:** Adam ir Nadam aplenkė RMSprop ir SGD
    - **Geriausias eksperimentas:** #24 (256-128-64-32, lr=0.0005, batch=64, dropout=0.3)
    """)
