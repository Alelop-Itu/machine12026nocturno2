# API de predicción experimental de cáncer de mama

**Autora:** Alejandra López · Desarrollo de Software, Instituto Superior Tecnológico Quito

API local hecha con **FastAPI** que usa una **neurona de McCulloch-Pitts** entrenada en el notebook `Neurona_McCulloch_y_Pitts.ipynb` con el dataset *Breast Cancer Wisconsin (Diagnostic)*.

> ⚠️ Proyecto educativo. Este modelo no sirve para diagnosticar ni descartar cáncer.

## Contenido de la carpeta

| Archivo | Para qué sirve |
|---|---|
| `Neurona_McCulloch_y_Pitts.ipynb` | Entrena y evalúa la neurona y genera `modelo_mp.joblib` |
| `modelo_mp.joblib` | Modelo entrenado que usa la API |
| `main.py` | API con FastAPI |
| `index.html` | Interfaz web sencilla para usar la API |
| `requirements.txt` | Librerías necesarias |

## Cómo ejecutarlo

Abre **Anaconda Prompt** dentro de esta carpeta.

**1. Instalar las librerías** (solo la primera vez)

```
pip install -r requirements.txt
```

**2. Generar el modelo**

Abre `Neurona_McCulloch_y_Pitts.ipynb` en Jupyter y ejecuta todas las celdas (*Kernel → Restart & Run All*). Al final se crea `modelo_mp.joblib` en esta misma carpeta.

> Si `modelo_mp.joblib` ya está en la carpeta, puedes saltar este paso.

**3. Levantar la API**

```
uvicorn main:app --reload
```

Deja esa ventana abierta mientras uses la API. Para detenerla, presiona `Ctrl + C`.

**4. Usarla**

- Interfaz web: http://127.0.0.1:8000
- Documentación interactiva: http://127.0.0.1:8000/docs

En la interfaz puedes cargar casos reales con un botón (benigno, maligno y un caso que el modelo falla) o escribir los 30 valores a mano.

## Endpoints

| Método | Ruta | Qué hace |
|---|---|---|
| GET | `/` | Interfaz web |
| GET | `/modelo` | Threshold y métricas del modelo |
| GET | `/caracteristicas` | Nombres de las 30 características y sus medianas |
| GET | `/ejemplos` | Casos reales del conjunto de prueba |
| GET | `/aleatorio` | Un paciente al azar del conjunto de prueba |
| POST | `/predecir` | Recibe las 30 características y devuelve la predicción |

## Resultados

| Métrica | Valor |
|---|---|
| Threshold óptimo | 27 de 30 entradas |
| Exactitud en prueba | 88.8 % (143 casos) |
| Casos malignos detectados | 73.6 % |

De 53 casos malignos, el modelo clasifica 14 como benignos (falsos negativos). Es la principal limitación de un modelo tan simple.

## Datos

Breast Cancer Wisconsin (Diagnostic), UCI Machine Learning Repository: https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic
