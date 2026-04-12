"""
database/models.py
SQLAlchemy modeliai - lentelių aprašymai
"""
from sqlalchemy import (
    Column, Integer, Float, String, Boolean,
    DateTime, Text, create_engine
)
from sqlalchemy.orm import declarative_base
from datetime import datetime

Base = declarative_base()


class Customer(Base):
    """Pagrindinė klientų lentelė su originaliais ir papildomais laukais."""
    __tablename__ = "customers"

    id                    = Column(Integer, primary_key=True, autoincrement=True)
    customer_id           = Column(String(50), unique=True, nullable=False)

    # Originalūs laukai
    gender                = Column(String(10))
    senior_citizen        = Column(Integer)          # 0 arba 1
    partner               = Column(String(5))
    dependents            = Column(String(5))
    tenure                = Column(Integer)
    phone_service         = Column(String(5))
    multiple_lines        = Column(String(20))
    internet_service      = Column(String(20))
    online_security       = Column(String(20))
    online_backup         = Column(String(20))
    device_protection     = Column(String(20))
    tech_support          = Column(String(20))
    streaming_tv          = Column(String(20))
    streaming_movies      = Column(String(20))
    contract              = Column(String(20))
    paperless_billing     = Column(String(5))
    payment_method        = Column(String(30))
    monthly_charges       = Column(Float)
    total_charges         = Column(Float)
    churn                 = Column(Integer)          # 0 arba 1

    # Išvestiniai (engineered) laukai
    tenure_group          = Column(String(20))       # "0-1 metai", "1-2 metai" ...
    charges_per_month_ratio = Column(Float)          # total_charges / tenure
    service_count         = Column(Integer)          # susumuotos paslaugos
    has_streaming         = Column(Integer)          # 0/1
    has_security_services = Column(Integer)          # 0/1
    is_long_term_contract = Column(Integer)          # Month-to-month = 0, kitaip 1
    avg_monthly_over_tenure = Column(Float)          # monthly_charges / (tenure+1)
    high_value_customer   = Column(Integer)          # monthly_charges > 70 → 1

    created_at            = Column(DateTime, default=datetime.utcnow)


class ModelResult(Base):
    """Modelių treniravimo rezultatų lentelė."""
    __tablename__ = "model_results"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    model_name      = Column(String(50))             # "RandomForest", "NeuralNetwork"
    experiment_id   = Column(String(50))             # unikalus eksperimento ID
    accuracy        = Column(Float)
    precision_score = Column(Float)
    recall          = Column(Float)
    f1              = Column(Float)
    roc_auc         = Column(Float)
    notes           = Column(Text)                   # hyperparametrai JSON eilutė
    trained_at      = Column(DateTime, default=datetime.utcnow)


class HyperparamExperiment(Base):
    """20–30 neuroninio tinklo eksperimentų lentelė."""
    __tablename__ = "hyperparam_experiments"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    experiment_no   = Column(Integer)
    layers          = Column(String(100))            # pvz. "128-64-32"
    neurons         = Column(String(100))
    learning_rate   = Column(Float)
    batch_size      = Column(Integer)
    optimizer       = Column(String(20))
    epochs          = Column(Integer)
    dropout         = Column(Float)
    activation      = Column(String(20))
    val_accuracy    = Column(Float)
    val_loss        = Column(Float)
    val_f1          = Column(Float)
    val_roc_auc     = Column(Float)
    notes           = Column(Text)
    trained_at      = Column(DateTime, default=datetime.utcnow)


class Prediction(Base):
    """Naudotojo gautos prognozės."""
    __tablename__ = "predictions"

    id              = Column(Integer, primary_key=True, autoincrement=True)
    model_used      = Column(String(50))
    input_data      = Column(Text)                   # JSON
    churn_probability = Column(Float)
    prediction      = Column(Integer)                # 0 arba 1
    predicted_at    = Column(DateTime, default=datetime.utcnow)
