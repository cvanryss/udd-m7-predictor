# 🧠 Predicción de Modalidad Laboral — Generación Z

> Proyecto del Módulo 7 · Bootcamp Ciencia de Datos e Inteligencia Artificial · UDD  
> **Cristian Van Rysselberghe · 2025**

---

## 📋 Descripción del Proyecto

Este proyecto construye un modelo de **Machine Learning** capaz de predecir la modalidad laboral preferida de jóvenes de la Generación Z a partir de sus respuestas a una encuesta de aspiraciones profesionales.

Las tres clases posibles son:

| Clase | Descripción |
|-------|-------------|
| 🏠 **Remote** | Trabajo 100% remoto |
| 🔀 **Hybrid** | Trabajo híbrido (presencial + remoto) |
| 🏢 **Onsite** | Trabajo 100% presencial |

---

## 📊 Dataset

- **Fuente:** [Career Aspirations of Gen Z — Kaggle](https://www.kaggle.com/datasets/kulturehire/understanding-career-aspirations-of-genz)
- **Registros:** ~235 encuestados
- **Variables originales:** 15
- **Variables usadas:** 13 (se eliminaron `Country` y `Zip Code` por no ser informativas)
- **Valores nulos:** 0 (dataset completo)
- **Distribución del target:**

```
Hybrid    → 117 casos
Remote    →  79 casos
Onsite    →  39 casos
```

---

## 🗂️ Estructura del Proyecto

```
├── Cristian_Van_Rysselberghe_UDD_Proyecto_M7.ipynb   # Notebook principal
├── modelo_udds_m7.joblib                              # Modelo entrenado exportado
├── app.py                                             # API REST con FastAPI
├── requirements.txt                                   # Dependencias
└── README.md
```

---

## ⚙️ Metodología

### 1. Análisis Exploratorio y Limpieza
- Revisión de tipos de datos, distribuciones y valores únicos
- Eliminación de `Your Current Country` (>98% India, sin variabilidad)
- Eliminación de `Your Current Zip Code` (cardinalidad extrema, sin valor predictivo)
- Reagrupación de 6 categorías de modalidad laboral → 3 clases (`Remote`, `Hybrid`, `Onsite`)

### 2. Preprocesamiento
- **One-Hot Encoding** para todas las variables categóricas (`sklearn.ColumnTransformer`)
- Variable numérica ordinal (escala 1–10) dejada sin escalamiento
- División **80/20** con estratificación para preservar proporciones de clases

### 3. Modelos Entrenados
| Modelo | Accuracy | F1-Macro |
|--------|----------|----------|
| Regresión Logística (baseline) | 38% | 0.36 |
| Random Forest | 51% | 0.46 |
| **Random Forest (tuned)** ✅ | **53%** | **0.47** |

> Random Forest corresponde a un modelo de ensamble (bagging) basado en múltiples árboles de decisión.

### 4. Optimización de Hiperparámetros
Se utilizó `GridSearchCV` con validación cruzada de 5 pliegues (`cv=5`) y métrica `f1_macro`:

```python
param_grid = {
    "model__n_estimators":     [200, 400],
    "model__max_depth":        [None, 10, 20],
    "model__min_samples_split":[2, 10],
    "model__min_samples_leaf": [1, 2],
}
```

### 5. Variables más Influyentes
Principales variables predictivas:
- Expectativa salarial
- Misión buscada en la empresa
- Tipo de empresa ideal
- Factor de elección de empleador
- Género

---

## 🚀 API REST

El modelo fue exportado con `joblib` y desplegado como API usando **FastAPI** en **Render**.

🔗 **URL pública:** [https://udd-m7-predictor.onrender.com/docs](https://udd-m7-predictor.onrender.com/docs)

La API retorna la predicción junto con las probabilidades asociadas a cada clase, lo que permite interpretar el nivel de confianza del modelo. La API permite enviar datos vía método POST y retorna una predicción junto con las probabilidades de cada clase, cumpliendo el requisito de exposición del modelo en la nube.

### Endpoint principal

```
POST https://udd-m7-predictor.onrender.com/predict
```

### Ejemplo de Request

```json
{
  "record": {
    "Your Gender": "Male",
    "What is your most preferred working environment?": "Mostly Remote",
    ...
  }
}
```

### Ejemplo de Response

```json
{
  "prediction": "Hybrid",
  "probabilities": {
    "Hybrid": 0.635,
    "Onsite": 0.255,
    "Remote": 0.110
  }
}
```

---

## 🛠️ Tecnologías Utilizadas

| Herramienta | Uso |
|-------------|-----|
| `pandas` | Carga y manipulación de datos |
| `scikit-learn` | Modelos ML, pipelines, métricas |
| `matplotlib` | Visualizaciones |
| `joblib` | Serialización del modelo |
| `FastAPI` | API REST |
| `pydantic` | Validación de datos en la API |
| `Render` | Despliegue en la nube de la API |
| `Google Colab` | Entorno de desarrollo |

---

## 📈 Métricas de Rendimiento

Las métricas finales del modelo seleccionado (**Random Forest tuned**) sobre el conjunto de test:

> Se utiliza **F1-macro como métrica principal** por tratarse de un problema multiclase con desbalance entre clases.

```
              precision    recall  f1-score   support

      Hybrid       0.59      0.54      0.56        24
      Onsite       0.44      0.57      0.50         7
      Remote       0.56      0.63      0.59        16

    accuracy                           0.53        47
   macro avg       0.53      0.58      0.55        47
weighted avg       0.55      0.53      0.54        47
```

> **Nota:** Un clasificador aleatorio entre 3 clases obtendría ~33% de accuracy. El modelo logra 53%, representando una mejora significativa considerando el tamaño reducido del dataset.

---

## 💡 Aprendizajes Clave

1. **La limpieza de datos supera la complejidad del modelo** — eliminar columnas irrelevantes y reagrupar categorías tuvo más impacto que probar algoritmos más sofisticados.
2. **Los ensambles capturan mejor las relaciones no lineales** — Random Forest superó a la Regresión Logística porque las preferencias laborales no siguen patrones lineales.
3. **El F1-Macro es más informativo que la Accuracy** cuando las clases están desbalanceadas.
4. **El tuning tiene límites** — la mejora del 2% post-Grid Search indica que el cuello de botella es la cantidad de datos (~235 filas), no los hiperparámetros.
5. **El contexto cultural importa** — el dataset es mayoritariamente indio; aplicarlo en otro contexto requeriría nuevos datos de entrenamiento.

---

## 🏃 Cómo Ejecutar

### Instalar dependencias

```bash
pip install fastapi uvicorn pyngrok joblib pandas scikit-learn
```

### Probar la API en producción

La API está desplegada y disponible públicamente en Render:

```
🔗 https://udd-m7-predictor.onrender.com/docs
```

Desde la documentación interactiva (Swagger UI) puedes probar los endpoints directamente en el navegador sin instalar nada.

### Ejecutar la API localmente

```bash
uvicorn app:app --reload
```

### Ejecutar el notebook

Abrir `Cristian_Van_Rysselberghe_UDD_Proyecto_M7.ipynb` en Google Colab o Jupyter y ejecutar las celdas en orden.

---

## 📄 Licencia

Proyecto académico desarrollado para el Bootcamp de Ciencia de Datos e IA de la Universidad del Desarrollo (UDD). Uso educativo.
