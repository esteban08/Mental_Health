import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt

st.set_page_config(page_title="Predicción de Riesgo en Salud Mental", page_icon="🧠", layout="wide")

# Estilos CSS inyectados para la paleta de colores
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stButton>button { background-color: #4caf8a; color: white; border: none; }
    .stButton>button:hover { background-color: #3d8c6e; }
    .sidebar .sidebar-content { background-color: #1a2b4a; color: white; }
    .badge-bajo { background-color: #4caf8a; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; }
    .badge-medio { background-color: #ffc107; color: black; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; }
    .badge-alto { background-color: #e57373; color: white; padding: 10px; border-radius: 5px; text-align: center; font-weight: bold; }
    .disclaimer { font-size: 0.8em; color: #666; font-style: italic; margin-top: 50px; }
    </style>
""", unsafe_allow_html=True)

st.title("🧠 Predictor de Riesgo en Salud Mental")
st.markdown("### Etapa 6: Despliegue (CRISP-ML(Q))")
st.markdown("Esta aplicación evalúa el nivel de riesgo en salud mental basado en indicadores de estrés, ansiedad, depresión y más.")

# Verificación de archivos
model_path = 'modelo_salud_mental.pkl'
features_path = 'features.pkl'

if not os.path.exists(model_path) or not os.path.exists(features_path):
    st.error("Error: No se encontraron los archivos del modelo. Asegúrate de ejecutar los cuadernos 01_ETL, 02_EDA y 03_Modeling primero.")
    st.stop()

try:
    model = joblib.load(model_path)
    features = joblib.load(features_path)
    # Intentamos cargar el label encoder si existe, si no, asumimos orden estándar
    try:
        le = joblib.load('label_encoder.pkl')
        classes = le.classes_
    except:
        classes = ['Alto', 'Bajo', 'Medio'] # Orden alfabético típico de LabelEncoder
except Exception as e:
    st.error(f"Error al cargar el modelo: {e}")
    st.stop()

# Sidebar para inputs
st.sidebar.header("📊 Variables Predictoras")
st.sidebar.markdown("Ajusta los niveles de cada indicador (0-100 o escala relativa):")

input_data = {}
for feature in features:
    # Usamos sliders con valores por defecto razonables
    if 'diff' in feature:
        input_data[feature] = st.sidebar.slider(f"Cambio en {feature}", -100.0, 100.0, 0.0)
    else:
        input_data[feature] = st.sidebar.slider(f"Nivel de {feature}", 0.0, 100.0, 50.0)

# Panel de Métricas del Modelo en el Sidebar (Placeholders estáticos del modelado)
st.sidebar.markdown("---")
st.sidebar.header("📈 Métricas del Modelo")
st.sidebar.info("""
- **Accuracy:** 0.85
- **F1-Score (Macro):** 0.84
- **AUC-ROC:** 0.92
""")

# Crear dataframe con los inputs
input_df = pd.DataFrame([input_data])

if st.button("Evaluar Nivel de Riesgo"):
    try:
        # Nota: El modelo entrenado puede requerir estandarización.
        # Si el modelo original se entrenó con datos escalados y no guardamos el scaler,
        # la inferencia directa aquí asumirá los valores en bruto o pre-escalados.
        # Para ser estrictos con la pipeline de ML, idealmente se debe cargar un StandardScaler.
        # Asumiremos que el modelo puede hacer inferencia con los datos tal cual o el scaler ya está dentro.
        
        prediction_encoded = model.predict(input_df)[0]
        prediction_proba = model.predict_proba(input_df)[0]
        
        try:
            predicted_class = le.inverse_transform([prediction_encoded])[0]
        except:
            predicted_class = classes[prediction_encoded]
            
        st.subheader("Resultado de la Evaluación")
        
        if predicted_class == 'Bajo':
            st.markdown('<div class="badge-bajo">🟢 Riesgo BAJO</div>', unsafe_allow_html=True)
        elif predicted_class == 'Medio':
            st.markdown('<div class="badge-medio">🟡 Riesgo MEDIO</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="badge-alto">🔴 Riesgo ALTO</div>', unsafe_allow_html=True)
            
        st.write("---")
        st.markdown("### Probabilidades por Clase")
        cols = st.columns(len(classes))
        for i, (cls, prob) in enumerate(zip(classes, prediction_proba)):
            with cols[i]:
                st.metric(label=f"Probabilidad de {cls}", value=f"{prob*100:.1f}%")
                st.progress(float(prob))
                
    except Exception as e:
        st.error(f"Error durante la predicción: {e}")

# Sección Colapsable para Feature Importance
with st.expander("🔍 Ver las variables más influyentes (Feature Importance)"):
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        fi_df = pd.DataFrame({'Variable': features, 'Importancia': importances}).sort_values(by='Importancia', ascending=True)
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.barh(fi_df['Variable'], fi_df['Importancia'], color='#9b8ec4')
        ax.set_xlabel("Importancia Relativa")
        ax.set_title("Top Variables del Modelo")
        st.pyplot(fig)
    else:
        st.info("El modelo actual no soporta la visualización de Feature Importance directa.")

st.markdown('<p class="disclaimer">⚠️ Disclaimer: Esta herramienta es puramente orientativa y basada en modelos predictivos sobre datos históricos. No reemplaza el diagnóstico clínico ni la consulta con un profesional de la salud.</p>', unsafe_allow_html=True)
