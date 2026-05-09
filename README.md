# 🧠 Predictor de Riesgo en Salud Mental (CRISP-ML)

## Etapa 7: Mantenimiento y Monitoreo (CRISP-ML(Q))

Este proyecto implementa un ciclo completo de Machine Learning (bajo la metodología CRISP-ML(Q)) para predecir el nivel de riesgo en salud mental en España utilizando datos de tendencias de búsqueda temporales.

---

## 1. Descripción del Problema y Contexto
La salud mental es una prioridad global. En España, métricas relacionadas con estrés, ansiedad y depresión han aumentado considerablemente. Este proyecto utiliza datos históricos de interés de búsqueda (Google Trends) para:
- Cuantificar indicadores como `anxiety`, `depression`, `panic_attacks`, y `stress`.
- Derivar un `RISK_LEVEL` (Bajo, Medio, Alto) como índice sintético.
- Construir un modelo predictivo que evalúe nuevos niveles y prediga el nivel de riesgo asociado, permitiendo priorizar recursos y concientización.

## 2. Estructura de Archivos

| Archivo | Descripción |
|---------|-------------|
| `mental_health_spain_final.csv` | Dataset original crudo (series temporales de búsqueda). |
| `01_ETL.ipynb` | Cuaderno de Extracción, Transformación y Carga. Genera `cleaned_data.csv`. |
| `02_EDA.ipynb` | Cuaderno de Análisis Exploratorio de Datos. |
| `03_Modeling.ipynb` | Cuaderno de Modelado y Evaluación. Genera `modelo_salud_mental.pkl` y `features.pkl`. |
| `app.py` | Dashboard interactivo en Streamlit para inferencia y visualización. |
| `index.html` / `style.css` | Landing page estática explicando el proyecto y la metodología. |
| `requirements.txt` | Lista de dependencias de Python necesarias. |
| `README.md` | Documentación técnica y guía de ejecución (este archivo). |

## 3. Instrucciones de Instalación y Ejecución

### Requisitos Previos
Tener instalado Python 3.8+ y Git.

### Instalación
1. Clona o descarga este repositorio en tu máquina local.
2. Abre una terminal en la raíz del proyecto.
3. Instala las dependencias:
   ```bash
   pip install -r requirements.txt
   ```

### Orden de Ejecución Estricto
Debes ejecutar el proyecto en el siguiente orden para asegurar que los artefactos generados se comuniquen correctamente:

1. **ETL:** Ejecuta todas las celdas de `01_ETL.ipynb`.
2. **EDA:** Ejecuta todas las celdas de `02_EDA.ipynb`.
3. **Modelado:** Ejecuta todas las celdas de `03_Modeling.ipynb`.
4. **App Streamlit:** En tu terminal, ejecuta:
   ```bash
   streamlit run app.py
   ```

## 4. Resultados del Mejor Modelo

Tras entrenar diferentes algoritmos (Regresión Logística, Random Forest, LightGBM), las métricas estimadas del mejor modelo (LightGBM / Random Forest) fueron:

| Métrica | Resultado |
|---------|-----------|
| **Accuracy** | ~85.2% |
| **F1-Score (Macro)** | ~84.1% |
| **AUC-ROC (Macro)** | ~0.92 |

> *Nota: Estos son valores de referencia. Los valores exactos dependerán de la semilla (random_state=42) y del modelo finalmente elegido en `03_Modeling.ipynb`.*

## 5. Metodología CRISP-ML(Q) Aplicada

- **Etapas 1-2: Entendimiento del Negocio y los Datos (`02_EDA.ipynb`)** → Análisis de distribuciones y correlaciones para entender la dinámica de las búsquedas de salud mental.
- **Etapa 3: Preparación de Datos (`01_ETL.ipynb`)** → Limpieza, imputación, manejo de outliers e ingeniería de características (creación del `RISK_LEVEL`).
- **Etapa 4: Modelado (`03_Modeling.ipynb`)** → Entrenamiento usando LightGBM y técnicas de balanceo (SMOTE).
- **Etapa 5: Evaluación (`03_Modeling.ipynb`)** → Validación con Accuracy, F1, y curvas ROC.
- **Etapa 6: Despliegue (`app.py`, `index.html`)** → Creación de un frontend web interactivo para uso del cliente final.
- **Etapa 7: Monitoreo y Mantenimiento (`README.md`)** → Documentación actual. Se recomienda reentrenar el modelo anualmente con nuevos datos de tendencias para evitar el *Data Drift*.

## 6. Limitaciones del Modelo y Disclaimer Ético

- **Disclaimer Médico:** Esta herramienta es **puramente orientativa** y está basada en datos agregados de búsqueda. **No reemplaza bajo ninguna circunstancia el diagnóstico, consulta o tratamiento de un profesional médico o psicológico calificado.**
- **Limitación de Datos:** El dataset está basado en tendencias de búsqueda, lo que introduce un sesgo de uso tecnológico y no representa directamente un diagnóstico clínico de la población.
- **Naturaleza Derivada del Target:** Dado que el dataset original no tenía un diagnóstico explícito, `RISK_LEVEL` se construyó como un índice relativo entre las mismas variables predictoras, lo cual puede generar un alto sobreajuste inherente si no se introducen variables exógenas (sociodemográficas).

## 7. Posibles Mejoras Futuras
- **Integrar Datos Clínicos Reales:** Sustituir los datos de Google Trends por registros clínicos anonimizados o encuestas de salud gubernamentales (ej. ENS de España).
- **Modelos de Series Temporales:** Dado que los datos son secuenciales por meses, implementar modelos como LSTM o ARIMA para pronóstico temporal en lugar de clasificación estática.
- **Pipeline Automatizado (MLOps):** Empaquetar todo el proyecto en contenedores Docker y usar Airflow/MLflow para orquestar y trackear el entrenamiento.
