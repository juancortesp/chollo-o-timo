# 🚗 ¿Chollo o timo? · Calculadora del precio justo de un coche de segunda mano en España

👉 **[Probar la calculadora](https://chollo-o-timo-coches.streamlit.app)** · 📄 **[Case study con vídeo demo](https://juancortesp.github.io/proyectos/coches/)** ([English](https://juancortesp.github.io/proyectos/coches/en/))

Introduces las características de un coche y el precio que piden en el anuncio, y la app te devuelve:
- el **precio justo estimado**,
- un **rango de precio normal**,
- una etiqueta: 🟢 **chollo** · 🟡 **precio normal** · 🔴 **caro**,
- y qué características suben o bajan el precio de ese coche.

---

## 1. El problema

Comprar un coche de segunda mano es difícil cuando no sabes si el precio de un anuncio es razonable. Este proyecto responde a dos preguntas:

1. **¿Qué variables determinan el precio** de un coche usado en España?
2. **¿Se puede estimar un precio justo** lo bastante preciso como para detectar anuncios baratos o caros?

## 2. Datos

| | |
|---|---|
| Fuente | Anuncios públicos de AutoScout24 España (scraping propio, 1-14 de octubre de 2026) |
| Tamaño final | **71.220 anuncios** tras la limpieza |
| Alcance | Turismos matriculados entre 2016 y 2026, precio entre 3.000 € y 150.000 €, sin km 0 |
| Variables | Marca, modelo, combustible, tipo de híbrido, cambio, carrocería, provincia, antigüedad, kilómetros, km/año y potencia |

Los datos se recogieron con fines educativos y de portfolio, sin uso comercial.

**Limitaciones**
- Son **precios pedidos**, no precios finales de venta.
- Es una **foto de octubre de 2026**: los precios cambian con el tiempo.
- Solo hay datos de **un portal**, con el perfil de vendedor de esa web.
- No hay **tipo de vendedor** (concesionario o particular), así que no se pudo estudiar su efecto.

## 3. Método

1. **Limpieza** (`00_preparacion_datos.ipynb`): normalización de marcas y modelos, traducción de categorías, eliminación de valores imposibles y **duplicados**. Incluye el mismo coche publicado en varias sedes de un concesionario: 15.576 anuncios, el 18 % del total.
2. **EDA** (`01_EDA.ipynb`): calidad de datos, distribuciones, outliers, relación de cada variable con el precio y contraste de hipótesis.
3. **Modelado** (`02_catboost_final.ipynb`):
   - Split 80/20 para train y test, más un 15 % de train para validación (`random_state=42`).
   - Imputación y encoding aprendidos **solo en train**.
   - Variable objetivo: `log(precio)`, por la asimetría del precio.
   - Modelos comparados sobre el mismo split: **baseline** (mediana por marca-modelo-edad), **Random Forest** y **CatBoost**.
4. **Calculadora** (`app.py`): Streamlit. El rango normal sale de los errores reales del modelo en test (percentiles 10 y 90).

## 4. Resultados

**Comparación de modelos (validación)**

| Modelo | Error medio (MAE) | Error % (MAPE) | R² |
|---|---|---|---|
| Baseline: mediana por marca, modelo y edad | 3.327 € | 13,1 % | 0,85 |
| Random Forest | 2.562 € | 9,7 % | 0,89 |
| **CatBoost** | **2.228 €** | **8,7 %** | **0,93** |

**Modelo final (CatBoost) en test, sobre 14.244 anuncios que nunca había visto:**
- Error medio: **2.118 € (8,5 %)**
- Explica el **92 %** de la variación del precio
- **Un 33 % menos de error** que comparar con coches del mismo modelo y edad

**Regla de la calculadora:** el 80 % de los anuncios piden entre un −12 % y un +14 % respecto al precio justo. Por debajo es 🟢 chollo; por encima, 🔴 caro.

## 5. Hallazgos principales

- **La depreciación no es lineal:** un coche pierde mucho valor en los primeros años y después la caída se suaviza.
- **La potencia es la variable que más pesa en el precio**, por delante de la antigüedad y los kilómetros. Resume la gama del coche.
- **El diésel vale algo menos que un coche equivalente de gasolina.** En bruto parece más caro, pero eso es porque los diésel suelen ser coches más grandes (paradoja de Simpson). Comparando el mismo modelo y la misma edad, sale ligeramente más barato.
- **La provincia apenas influye:** una vez conocido el coche, saber dónde se vende no mejora la predicción.
- **El cambio automático** se asocia a precios claramente más altos, aunque en parte es porque los automáticos suelen ser coches más nuevos y potentes.

> Son relaciones y predicciones, no causalidad demostrada.

## 6. Estructura del repositorio

```
chollo-o-timo/
├── python/
│   ├── 00_preparacion_datos.ipynb
│   ├── 01_EDA.ipynb
│   ├── 02_catboost_final.ipynb
│   ├── app.py                  ← calculadora Streamlit
│   ├── requirements.txt
│   └── modelos/
│       ├── catboost_final.cbm  ← modelo entrenado
│       └── config_app.json     ← rango y opciones de la app
├── dataset/                    ← datos limpios
└── data_raw/                   ← datos originales
```

## 7. Cómo ejecutarlo en local

```bash
git clone https://github.com/juancortesp/chollo-o-timo.git
cd chollo-o-timo/python
pip install -r requirements.txt
streamlit run app.py
```

## 8. Herramientas

Python · pandas · NumPy · scikit-learn · CatBoost · seaborn · matplotlib · Streamlit

---

**Juan Cortés Paredes** · Data Analyst
🌐 [Portfolio](https://juancortesp.github.io/) · [LinkedIn](https://www.linkedin.com/in/juan-cortes-paredes/) · [GitHub](https://github.com/juancortesp)
