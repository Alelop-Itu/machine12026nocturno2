"""
API de predicción experimental de cáncer de mama
Modelo: neurona de McCulloch-Pitts
Autora: Alejandra López

Ejecutar con:  uvicorn main:app --reload
"""
import random
from pathlib import Path

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

CARPETA = Path(__file__).resolve().parent
RUTA_MODELO = CARPETA / "modelo_mp.joblib"
RUTA_HTML = CARPETA / "index.html"

ADVERTENCIA = (
    "Resultado experimental con fines educativos. "
    "No sirve para diagnosticar ni descartar cáncer. "
    "Ante cualquier duda, consulta a un profesional de salud."
)

# Cargar el modelo generado por entrenar.py
if not RUTA_MODELO.exists():
    raise RuntimeError(
        "No se encontró modelo_mp.joblib. Ejecuta primero: python entrenar.py"
    )

modelo = joblib.load(RUTA_MODELO)
NOMBRES = modelo["caracteristicas"]
CORTES = np.array([modelo["puntos_corte"][n] for n in NOMBRES])
THRESHOLD = modelo["threshold"]

app = FastAPI(
    title="API de predicción experimental de cáncer de mama",
    description="Neurona de McCulloch-Pitts entrenada con el dataset Breast Cancer Wisconsin.",
    version="1.0",
)


EJEMPLOS = modelo.get("ejemplos", {})
CASOS_PRUEBA = modelo.get("casos_prueba", [])
EJEMPLO_DOCS = (
    {"datos": EJEMPLOS["benigno_acertado"]["datos"]}
    if "benigno_acertado" in EJEMPLOS
    else {"datos": modelo["medianas"]}
)


class DatosCancer(BaseModel):
    datos: dict[str, float]

    # Este ejemplo aparece ya cargado en /docs al pulsar "Try it out"
    model_config = {"json_schema_extra": {"examples": [EJEMPLO_DOCS]}}


@app.get("/")
def inicio():
    if RUTA_HTML.exists():
        return FileResponse(RUTA_HTML)
    return {
        "mensaje": "La API está funcionando. Prueba los endpoints en /docs"
    }


@app.get("/caracteristicas")
def caracteristicas():
    return {
        "caracteristicas": NOMBRES,
        "medianas": modelo["medianas"],
    }


@app.get("/modelo")
def info_modelo():
    return {
        "tipo": "Neurona de McCulloch-Pitts",
        "threshold": THRESHOLD,
        "total_caracteristicas": len(NOMBRES),
        "metricas": modelo.get("metricas", {}),
        "advertencia": ADVERTENCIA,
    }


@app.get("/ejemplos")
def ejemplos():
    """Casos reales del conjunto de prueba, listos para copiar en /predecir."""
    return EJEMPLOS


@app.get("/aleatorio")
def caso_aleatorio():
    """Devuelve un paciente al azar del conjunto de prueba."""
    if not CASOS_PRUEBA:
        raise HTTPException(
            status_code=404,
            detail="No hay casos de prueba. Ejecuta de nuevo el notebook para regenerar modelo_mp.joblib.",
        )
    i = random.randrange(len(CASOS_PRUEBA))
    return {"numero": i + 1, "total": len(CASOS_PRUEBA), **CASOS_PRUEBA[i]}


@app.post("/predecir")
def predecir(entrada: DatosCancer):
    datos = entrada.datos

    faltantes = [n for n in NOMBRES if n not in datos]
    if faltantes:
        raise HTTPException(
            status_code=400,
            detail=f"Faltan características: {', '.join(faltantes)}",
        )

    valores = np.array([datos[n] for n in NOMBRES], dtype=float)

    if not np.all(np.isfinite(valores)):
        raise HTTPException(
            status_code=400, detail="Ingresa valores numéricos válidos."
        )

    if np.any(valores < 0):
        raise HTTPException(
            status_code=400, detail="Las características no pueden ser negativas."
        )

    # Binarizar con los mismos cortes del entrenamiento y sumar
    binarios = (valores <= CORTES).astype(int)
    suma = int(binarios.sum())
    prediccion = int(suma >= THRESHOLD)

    return {
        "prediccion": prediccion,
        "resultado": "Benigno" if prediccion == 1 else "Maligno",
        "entradas_binarias": dict(zip(NOMBRES, binarios.tolist())),
        "suma_entradas": suma,
        "threshold": THRESHOLD,
        "advertencia": ADVERTENCIA,
    }
