"""
¿Chollo o timo? · Calculadora del precio justo de un coche de segunda mano en España
Modelo: CatBoost entrenado con ~71.000 anuncios de AutoScout24 (octubre 2026)
"""

import json
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor, Pool

# ---------------------------------------------------------------
# 1. Cargar modelo y configuración (solo una vez, se guarda en caché)
# ---------------------------------------------------------------
@st.cache_resource
def cargar_modelo():
    modelo = CatBoostRegressor()
    modelo.load_model("modelos/catboost_final.cbm")
    return modelo


@st.cache_data
def cargar_config():
    with open("modelos/config_app.json", encoding="utf-8") as f:
        return json.load(f)


modelo = cargar_modelo()
cfg = cargar_config()

COLUMNAS = cfg["columnas"]
CAT_COLS = ["make", "model_group", "fuel", "tipo_hibrido", "body_grupo", "provincia"]

# Nombres legibles para la explicación
NOMBRES = {
    "make": "Marca", "model_group": "Modelo", "fuel": "Combustible",
    "tipo_hibrido": "Tipo de híbrido", "es_automatico": "Cambio",
    "body_grupo": "Carrocería", "provincia": "Provincia",
    "antiguedad": "Antigüedad", "mileage_km": "Kilómetros",
    "km_anio": "Km por año", "power_hp": "Potencia",
}

# ---------------------------------------------------------------
# 2. Cabecera
# ---------------------------------------------------------------
st.set_page_config(page_title="¿Chollo o timo?", page_icon="🚗", layout="centered")
st.title("🚗 ¿Chollo o timo?")
st.write("Introduce los datos del anuncio y te digo si el precio es justo.")

# ---------------------------------------------------------------
# 3. Formulario
# ---------------------------------------------------------------
marcas = sorted(cfg["modelos_por_marca"].keys())

col1, col2 = st.columns(2)
with col1:
    marca = st.selectbox("Marca", marcas, index=marcas.index("SEAT") if "SEAT" in marcas else 0)
    modelo_coche = st.selectbox("Modelo", cfg["modelos_por_marca"][marca])
    combustible = st.selectbox("Combustible", cfg["fuel"])
    hibrido = st.selectbox("Tipo de híbrido", cfg["tipo_hibrido"],
                           help="MHEV = microhíbrido · HEV = híbrido · PHEV = enchufable")
    cambio = st.radio("Cambio", ["Manual", "Automático"], horizontal=True)
with col2:
    carroceria = st.selectbox("Carrocería", cfg["body_grupo"])
    provincia = st.selectbox("Provincia", cfg["provincia"])
    hoy = date.today()
    anio = st.number_input("Año de matriculación", min_value=2016, max_value=hoy.year,
                           value=2020, step=1)
    mes = st.number_input("Mes de matriculación", min_value=1, max_value=12, value=6, step=1)
    km = st.number_input("Kilómetros", min_value=1_000, max_value=500_000,
                         value=80_000, step=1_000)
    cv = st.number_input("Potencia (CV)", min_value=40, max_value=700, value=130, step=5)

precio_anuncio = st.number_input("💶 Precio que piden en el anuncio (€)", min_value=1_000,
                                 max_value=200_000, value=15_000, step=100)

# ---------------------------------------------------------------
# 4. Calcular
# ---------------------------------------------------------------
if st.button("Calcular precio justo", type="primary", use_container_width=True):

    # Variables derivadas, igual que en el entrenamiento
    antiguedad = max((hoy.year + (hoy.month - 1) / 12) - (anio + (mes - 1) / 12), 0)
    km_anio = km / max(antiguedad, 0.5)

    fila = pd.DataFrame([{
        "make": marca, "model_group": modelo_coche, "fuel": combustible,
        "tipo_hibrido": hibrido, "es_automatico": int(cambio == "Automático"),
        "body_grupo": carroceria, "provincia": provincia,
        "antiguedad": antiguedad, "mileage_km": km, "km_anio": km_anio, "power_hp": cv,
    }])[COLUMNAS]

    # Predicción (el modelo trabaja en log del precio)
    log_pred = modelo.predict(fila)[0]
    precio_justo = np.exp(log_pred)
    minimo = precio_justo * np.exp(cfg["p10"])
    maximo = precio_justo * np.exp(cfg["p90"])

    # Etiqueta
    if precio_anuncio < minimo:
        etiqueta, color = "🟢 CHOLLO", "green"
    elif precio_anuncio > maximo:
        etiqueta, color = "🔴 CARO", "red"
    else:
        etiqueta, color = "🟡 PRECIO NORMAL", "orange"

    dif = precio_anuncio - precio_justo
    dif_pct = dif / precio_justo

    # Resultado
    st.divider()
    st.markdown(f"## :{color}[{etiqueta}]")
    c1, c2, c3 = st.columns(3)
    c1.metric("Precio justo", f"{precio_justo:,.0f} €".replace(",", "."))
    c2.metric("Rango normal", f"{minimo:,.0f} – {maximo:,.0f} €".replace(",", "."))
    c3.metric("El anuncio pide", f"{precio_anuncio:,.0f} €".replace(",", "."),
              delta=f"{dif:+,.0f} € ({dif_pct:+.0%})".replace(",", "."), delta_color="inverse")

    # Explicación: qué sube y qué baja el precio de ESTE coche (valores SHAP)
    pool = Pool(fila, cat_features=CAT_COLS)
    shap = modelo.get_feature_importance(pool, type="ShapValues")[0][:-1]  # última = base
    efectos = (pd.Series(np.exp(shap) - 1, index=COLUMNAS)
               .rename(index=NOMBRES)
               .sort_values(key=abs, ascending=False)
               .head(5))

    st.subheader("¿Qué mueve el precio de este coche?")
    st.caption("Comparado con un coche medio del mercado")
    for nombre, efecto in efectos.items():
        flecha = "⬆️" if efecto > 0 else "⬇️"
        st.write(f"{flecha} **{nombre}**: {efecto:+.0%}")

    if etiqueta.endswith("CHOLLO"):
        st.warning("Un precio muy por debajo puede ser una oportunidad… o esconder algo "
                   "(accidente, averías, cargas). Revisa el historial antes de comprar.")

# ---------------------------------------------------------------
# 5. Pie
# ---------------------------------------------------------------
st.divider()
st.caption(
    f"Modelo CatBoost entrenado con ~71.000 anuncios de AutoScout24 España (octubre 2026). "
    f"Error medio en test: {cfg['mae_test']:,} € (8,5 %). ".replace(",", ".") +
    "Es una estimación sobre precios pedidos, no precios finales de venta. "
    "Proyecto de portfolio de Juan · Data Analyst."
)
