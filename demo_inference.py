"""
================================================================================
PROYECTO PRÁCTICO: REDES NEURONALES - UNIVERSIDAD DEL VALLE (2026-2)
ENTREGA 1: DEMO INTERACTIVA DE INFERENCIA EN VIVO
DATASET 3: 20 NEWSGROUPS (PERCEPTRÓN MULTICAPA / KERAS)
================================================================================
Este script permite realizar inferencias en tiempo real sobre textos nuevos
ingresados por consola o pasados por argumento. Cumple con el requerimiento
de 'Demo Ejecutable' exigido para la sustentación oral (Rúbrica de evaluación).
"""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path

import numpy as np

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
from sklearn.datasets import fetch_20newsgroups


# Categorías temáticas agrupadas para mejor visualización
CATEGORY_TOPICS = {
    "comp": "Computación y Sistemas",
    "rec": "Recreación, Deportes y Vehículos",
    "sci": "Ciencia y Medicina",
    "talk": "Sociedad, Política y Religión",
    "misc": "Varios",
    "alt": "Alternativo / Filosofía",
}


def load_artifacts(model_path: str, vectorizer_path: str):
    """Carga el modelo Keras y el vectorizador TF-IDF serializado."""
    m_path = Path(model_path)
    v_path = Path(vectorizer_path)

    if not m_path.exists():
        raise FileNotFoundError(f"No se encontró el modelo en: {m_path.resolve()}")
    if not v_path.exists():
        raise FileNotFoundError(f"No se encontró el vectorizador en: {v_path.resolve()}")

    print(f"Cargando modelo Keras desde:       {m_path.resolve()}")
    model = keras.models.load_model(m_path)

    print(f"Cargando vectorizador TF-IDF desde: {v_path.resolve()}")
    with open(v_path, "rb") as f:
        vectorizer = pickle.load(f)

    # Obtenemos los nombres oficiales de las 20 clases
    dataset_info = fetch_20newsgroups(subset="train")
    class_names = dataset_info.target_names

    return model, vectorizer, class_names


def predict_text(
    text: str,
    model: keras.Model,
    vectorizer,
    class_names: list[str],
    top_k: int = 5,
) -> None:
    """Clasifica un texto e imprime las predicciones con formato visual."""
    X = vectorizer.transform([text]).toarray()
    probs = model.predict(X, verbose=0)[0]

    top_indices = np.argsort(probs)[-top_k:][::-1]

    print("\n" + "=" * 65)
    print("ANÁLISIS DE CLASIFICACIÓN DE TEXTO")
    print("=" * 65)
    print(f"Texto ingresado:\n\"{text.strip()}\"\n")
    print(f"{'Ranking':<8}{'Categoría':<28}{'Confianza':<12}{'Barra'}")
    print("-" * 65)

    for rank, idx in enumerate(top_indices, start=1):
        cname = class_names[idx]
        conf = probs[idx] * 100
        bar = "#" * int(conf / 4)
        print(f"#{rank:<7}{cname:<28}{conf:6.2f}%    {bar}")

    best_cat = class_names[top_indices[0]]
    domain_prefix = best_cat.split(".")[0]
    domain_label = CATEGORY_TOPICS.get(domain_prefix, "General")

    print("-" * 65)
    print(f">> Categoria Predicha: {best_cat}")
    print(f">> Dominio Tematico:   {domain_label}")
    print("=" * 65 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Demo Ejecutable para Sustentación (20 Newsgroups MLP)")
    parser.add_argument("--model-path", type=str, default="models/baseline_mlp.keras", help="Ruta al modelo .keras")
    parser.add_argument("--vectorizer-path", type=str, default="models/vectorizer_baseline_mlp.pkl", help="Ruta al vectorizer .pkl")
    parser.add_argument("--text", type=str, default=None, help="Texto directo en inglés a clasificar")
    parser.add_argument("--top-k", type=int, default=5, help="Número de predicciones a mostrar")
    parser.add_argument("--run-samples", action="store_true", help="Ejecuta los 5 casos de prueba (2 claros y 3 mixtos)")
    args = parser.parse_args()

    # Si no existe baseline_mlp, intentamos con test_run si existe
    m_path = args.model_path
    v_path = args.vectorizer_path
    if not os.path.exists(m_path) and os.path.exists("models/test_run.keras"):
        m_path = "models/test_run.keras"
        v_path = "models/vectorizer_test_run.pkl"

    model, vectorizer, class_names = load_artifacts(m_path, v_path)

    if args.run_samples:
        preset_samples = [
            ("NASA rover detects evidence of liquid water and atmosphere on Mars.", "CASO CLARO (Alta certeza)"),
            ("The engine needs a new oil filter and spark plugs for better mileage.", "CASO CLARO (Alta certeza)"),
            ("A debate on whether doctors should perform experimental medical treatments when religious beliefs oppose it.", "CASO MIXTO (Medicina vs. Religión/Ética)"),
            ("I installed a new operating system on my PC to play baseball and hockey video games.", "CASO MIXTO (Sistemas vs. Deportes)"),
            ("Congressional debate on government funding for public healthcare hospitals and health insurance.", "CASO MIXTO (Política vs. Salud Pública)"),
        ]
        print("\n" + "=" * 65)
        print("  EJECUTANDO MUESTRAS PRECONFIGURADAS (2 CLARAS + 3 MIXTAS)")
        print("=" * 65)
        for text, category in preset_samples:
            print(f"\n>>> [{category}]")
            predict_text(text, model, vectorizer, class_names, top_k=args.top_k)
        return

    if args.text:
        predict_text(args.text, model, vectorizer, class_names, top_k=args.top_k)
        return

    # Modo interactivo en bucle
    print("\n" + "=" * 65)
    print("  DEMO INTERACTIVA EN TIEMPO REAL - CLASIFICACIÓN DE NOTICIAS")
    print("  Escribe un fragmento de texto en inglés y presiona ENTER.")
    print("  Escribe 'salir' o 'exit' para terminar.")
    print("=" * 65)

    while True:
        try:
            user_input = input("\nIngresa texto a clasificar > ").strip()
            if not user_input:
                continue
            if user_input.lower() in {"salir", "exit", "quit"}:
                print("Finalizando sesión de demo. ¡Éxitos en la sustentación!")
                break
            predict_text(user_input, model, vectorizer, class_names, top_k=args.top_k)
        except (KeyboardInterrupt, EOFError):
            print("\nDemo finalizada.")
            break


if __name__ == "__main__":
    main()
