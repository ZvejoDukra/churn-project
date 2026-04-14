"""
utils/data_loader.py
CSV įkėlimas į DB ir visų transformacijų logika

"""
import pandas as pd
from sqlalchemy import select, func
from sqlalchemy.orm import Session
from database.models import Customer
from database.connection import init_db, get_session

REQUIRED_COLUMNS = ["customerID", "tenure", "MonthlyCharges", "TotalCharges", "Churn"]


# ---------------------------------------------------------------------------
# 1. Validacija
# ---------------------------------------------------------------------------

def validate_columns(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Patikrina ar DataFrame turi visus privalomus stulpelius
    Grąžina (True, "") jei viskas ok, arba (False, klaidos žinutė)
    """
    missing = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        return False, f"Trūksta stulpelių: {missing}"
    return True, ""


# ---------------------------------------------------------------------------
# 2. Transformacijos (feature engineering)
# ---------------------------------------------------------------------------

def _tenure_group(tenure: int) -> str:
    """Priskiria kliento stažo grupę pagal mėnesių skaič"""
    if tenure <= 12:
        return "0-1 metai"
    elif tenure <= 24:
        return "1-2 metai"
    elif tenure <= 48:
        return "2-4 metai"
    else:
        return "4+ metai"


def _count_services(row: pd.Series) -> int:
    """Suskaičiuoja kiek paslaugų klientas naudoja"""
    service_cols = [
        "PhoneService", "MultipleLines", "OnlineSecurity",
        "OnlineBackup", "DeviceProtection", "TechSupport",
        "StreamingTV", "StreamingMovies"
    ]
    count = 0
    for col in service_cols:
        if col in row and str(row[col]).strip().lower() not in (
            "no", "no phone service", "no internet service", "nan"
        ):
            count += 1
    return count


def apply_feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prideda 8 išvestinius stulpelius prie DataFrame
    Visi originalūs stulpeliai paliekami nepakeisti
    """
    df = df.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    df["tenure_group"]            = df["tenure"].apply(_tenure_group)
    df["charges_per_month_ratio"] = (df["TotalCharges"] / (df["tenure"] + 1)).round(4)
    df["service_count"]           = df.apply(_count_services, axis=1)
    df["has_streaming"]           = df.apply(
        lambda r: 1 if str(r.get("StreamingTV", "")).lower() == "yes"
                     or str(r.get("StreamingMovies", "")).lower() == "yes" else 0, axis=1)
    df["has_security_services"]   = df.apply(
        lambda r: 1 if str(r.get("OnlineSecurity", "")).lower() == "yes"
                     or str(r.get("TechSupport", "")).lower() == "yes" else 0, axis=1)
    df["is_long_term_contract"]   = df["Contract"].apply(
        lambda x: 0 if str(x).strip() == "Month-to-month" else 1)
    df["avg_monthly_over_tenure"] = (df["MonthlyCharges"] / (df["tenure"] + 1)).round(4)
    df["high_value_customer"]     = (df["MonthlyCharges"] > 70).astype(int)
    return df


def encode_churn(df: pd.DataFrame) -> pd.DataFrame:
    """
    Paverčia Churn stulpelį iš Yes/No tekstų į 1/0 skaičius ML mod
    """
    df = df.copy()
    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0, 1: 1, 0: 0}).fillna(0).astype(int)
    return df


# ---------------------------------------------------------------------------
# 3. Pagalbinės funkcijos
# ---------------------------------------------------------------------------

def customer_exists(session: Session, customer_id: str) -> bool:
    """
    Patikrina ar klientas su nurodytu ID jau egzistuoja DB
    """
    stmt = select(func.count()).select_from(Customer).where(
        Customer.customer_id == customer_id
    )
    count = session.execute(stmt).scalar()
    return count > 0


def build_customer_object(row: pd.Series) -> Customer:
    """
    Iš vienos DataFrame eilutės sukuria Customer ORM objektą
    Grąžina Customer instanciją paruoštą session.add() operacijai
    """
    return Customer(
        customer_id             = str(row["customerID"]),
        gender                  = str(row.get("gender", "")),
        senior_citizen          = int(row.get("SeniorCitizen", 0)),
        partner                 = str(row.get("Partner", "")),
        dependents              = str(row.get("Dependents", "")),
        tenure                  = int(row.get("tenure", 0)),
        phone_service           = str(row.get("PhoneService", "")),
        multiple_lines          = str(row.get("MultipleLines", "")),
        internet_service        = str(row.get("InternetService", "")),
        online_security         = str(row.get("OnlineSecurity", "")),
        online_backup           = str(row.get("OnlineBackup", "")),
        device_protection       = str(row.get("DeviceProtection", "")),
        tech_support            = str(row.get("TechSupport", "")),
        streaming_tv            = str(row.get("StreamingTV", "")),
        streaming_movies        = str(row.get("StreamingMovies", "")),
        contract                = str(row.get("Contract", "")),
        paperless_billing       = str(row.get("PaperlessBilling", "")),
        payment_method          = str(row.get("PaymentMethod", "")),
        monthly_charges         = float(row.get("MonthlyCharges", 0)),
        total_charges           = float(row.get("TotalCharges", 0)),
        churn                   = int(row["Churn"]),
        tenure_group            = str(row["tenure_group"]),
        charges_per_month_ratio = float(row["charges_per_month_ratio"]),
        service_count           = int(row["service_count"]),
        has_streaming           = int(row["has_streaming"]),
        has_security_services   = int(row["has_security_services"]),
        is_long_term_contract   = int(row["is_long_term_contract"]),
        avg_monthly_over_tenure = float(row["avg_monthly_over_tenure"]),
        high_value_customer     = int(row["high_value_customer"]),
    )


# ---------------------------------------------------------------------------
# 4. CSV  į SQLite
# ---------------------------------------------------------------------------

def load_csv_to_db(filepath: str) -> tuple[bool, str]:
    """
    Įkelia CSV failą į SQLite duomenų bazę
    Grąžina (sėkmė: bool, žinutė: str)
    """
    try:
        init_db()
        df = pd.read_csv(filepath)

        valid, error_msg = validate_columns(df)
        if not valid:
            return False, error_msg

        df = apply_feature_engineering(df)
        df = encode_churn(df)

        session: Session = get_session()
        added = 0

        for _, row in df.iterrows():
            if customer_exists(session, str(row["customerID"])):
                continue
            session.add(build_customer_object(row))
            added += 1

        session.commit()
        session.close()
        return True, f"Sėkmingai įkelta {added} klientų į DB."

    except Exception as e:
        return False, f"Klaida įkeliant duomenis: {e}"


# ---------------------------------------------------------------------------
# 5. DB → DataFrame (treniravimui)
# ---------------------------------------------------------------------------

def load_data_from_db() -> pd.DataFrame:
    """
    Grąžina visus klientus iš DB kaip DataFrame
    """
    session = get_session()
    stmt = select(Customer)
    customers = session.execute(stmt).scalars().all()
    session.close()

    if not customers:
        raise ValueError("Duomenų bazė tuščia. Pradžioje įkelti CSV")

    records = [{
        "tenure":                  c.tenure,
        "monthly_charges":         c.monthly_charges,
        "total_charges":           c.total_charges,
        "senior_citizen":          c.senior_citizen,
        "service_count":           c.service_count,
        "has_streaming":           c.has_streaming,
        "has_security_services":   c.has_security_services,
        "is_long_term_contract":   c.is_long_term_contract,
        "charges_per_month_ratio": c.charges_per_month_ratio,
        "avg_monthly_over_tenure": c.avg_monthly_over_tenure,
        "high_value_customer":     c.high_value_customer,
        "churn":                   c.churn,
    } for c in customers]

    return pd.DataFrame(records)
