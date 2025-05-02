
import streamlit as st
import pandas as pd
import numpy as np
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Predicción de Incendios Forestales", layout="centered")
st.title("🔥 Predicción de Incendios Forestales")
st.markdown("Este modelo predice si es probable que ocurra un incendio forestal en Santa Cruz o Beni (Bolivia), basado en condiciones climáticas y del terreno.")

@st.cache_resource
def load_model():
    df = pd.read_csv("dataset_incendios_santacruz_beni.csv")
    encoders = {}
    for col in ['Índice de sequía', 'Cobertura vegetal', 'Uso de suelo', 'Región']:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le
    df['Incendio'] = df['Incendio'].map({'No': 0, 'Sí': 1})
    X = df.drop(columns=['Fecha', 'Incendio'])
    y = df['Incendio']
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X, y)
    return model, encoders

model, encoders = load_model()

st.header("🌿 Ingrese los datos del entorno")

with st.form("form_pred"):
    col1, col2 = st.columns(2)
    with col1:
        lat = st.number_input("Latitud", value=-17.5)
        lon = st.number_input("Longitud", value=-63.0)
        temp = st.slider("Temperatura (°C)", 10, 45, 36)
        humidity = st.slider("Humedad (%)", 10, 100, 28)
    with col2:
        wind = st.slider("Viento (km/h)", 0, 40, 12)
        rain = st.slider("Precipitación (mm)", 0, 100, 2)
        drought = st.selectbox("Índice de sequía", ["Bajo", "Moderado", "Alto"])
        veg = st.selectbox("Cobertura vegetal", ["Bosque seco", "Sabana", "Agrícola", "Pasto"])
        land = st.selectbox("Uso de suelo", ["Agrícola", "Ganadero", "Forestal", "Urbano"])
        region = st.selectbox("Región", ["Santa Cruz", "Beni"])

    submit = st.form_submit_button("🔍 Predecir")

if submit:
    sample = pd.DataFrame([{
        "Latitud": lat,
        "Longitud": lon,
        "Temperatura": temp,
        "Humedad": humidity,
        "Viento": wind,
        "Precipitación": rain,
        "Índice de sequía": encoders["Índice de sequía"].transform([drought])[0],
        "Cobertura vegetal": encoders["Cobertura vegetal"].transform([veg])[0],
        "Uso de suelo": encoders["Uso de suelo"].transform([land])[0],
        "Región": encoders["Región"].transform([region])[0]
    }])
    prediction = model.predict(sample)[0]
    prob = model.predict_proba(sample)[0][1]
    if prediction == 1:
        st.error(f"⚠️ Alta probabilidad de incendio forestal ({prob*100:.1f}%)")
    else:
        st.success(f"✅ Baja probabilidad de incendio forestal ({prob*100:.1f}%)")
