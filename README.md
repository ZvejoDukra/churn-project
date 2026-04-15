# 📊 Klientų išėjimo (Churn) prognozavimo sistema

Baigiamasis darbas – telekomunikacijų klientų churn prognozavimas naudojant Random Forest ir Feed Forward neuroninį tinklą.

## 🗂 Projekto struktūra

```
churn_project/
├── app.py                        # Pagrindinis Streamlit frontentas
├── requirements.txt
├── data/                         # SQLite DB ir modelių failai (sukuriami automatiškai)
├── database/
│   ├── __init__.py
│   ├── models.py                 # SQLAlchemy ORM modeliai (lentelės)
│   └── connection.py             # DB ryšys ir pagalbinės f-jos
├── models/
│   ├── __init__.py
│   ├── random_forest_model.py    # Random Forest klasė
│   └── neural_network_model.py   # Feed Forward NN + 25 eksperimentai
└── utils/
    ├── __init__.py
    ├── data_loader.py            # CSV → DB + feature engineering
    ├── preprocessing.py          # Skalėjimas, SMOTE, duomenų paruošimas
    ├── metrics.py                # Metrikų skaičiavimas ir išsaugojimas
    └── visualizations.py         # 7 Plotly grafikai
```

## 🚀 Paleidimas (macOS)

```bash
# 1. Klonuokite repozitoriją
git clone <jūsų-repo-url>
cd churn_project

# 2. Sukurkite virtualią aplinką
python3 -m venv venv
source venv/bin/activate

# 3. Įdiekite priklausomybes
pip install -r requirements.txt

# 4. Paleiskite programą
streamlit run app.py
```

## 📋 Naudojimas

1. **Duomenų įkėlimas** – įkelkite `WA_Fn-UseC_-Telco-Customer-Churn.csv` iš Kaggle
2. **Modelių treniravimas** – pasirinkite Random Forest arba NN, spauskite "Trenuoti"
3. **Prognozavimas** – įveskite kliento duomenis ir gaukite prognozę
4. **Grafikai** – peržiūrėkite 4+ vizualizacijas
5. **Eksperimentai** – matykite visų 25 NN konfigūracijų rezultatus
6. **Išvados** – palyginkite modelius ir skaitykite tekstines išvadas

## 🗄️ Duomenų bazė

Naudojama SQLite per SQLAlchemy ORM. DB failo vieta: `data/churn.db`

**Lentelės:**
- `customers` – klientai + feature engineering stulpeliai
- `model_results` – apmokymo rezultatai (accuracy, F1, ROC-AUC ir kt.)
- `hyperparam_experiments` – 25 NN eksperimentų rezultatai
- `predictions` – vartotojo prognozės

## 📊 Feature Engineering (papildomi požymiai)

| Požymis | Aprašas |
|---------|---------|
| `tenure_group` | Stažo grupė (0-1, 1-2, 2-4, 4+ metai) |
| `charges_per_month_ratio` | Visos išlaidos / stažas |
| `service_count` | Naudojamų paslaugų kiekis |
| `has_streaming` | Ar turi srautinį turinį |
| `has_security_services` | Ar turi saugumo paslaugas |
| `is_long_term_contract` | 0 = mėnesinis, 1 = ilgalaikis |
| `avg_monthly_over_tenure` | Mėn. išlaidos / (stažas+1) |
| `high_value_customer` | monthly_charges > 70 → 1 |

## 🧪 Modeliai

- **Random Forest** – scikit-learn, class_weight="balanced", SMOTE
- **Neuroninis tinklas** – TensorFlow/Keras, Feed Forward, 25 eksperimentai
