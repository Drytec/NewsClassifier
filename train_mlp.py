"""
================================================================================
PROYECTO PRÁCTICO: REDES NEURONALES - UNIVERSIDAD DEL VALLE (2026-2)
ENTREGA 1: CLASIFICACIÓN DE TEXTO CON PERCEPTRÓN MULTICAPA (MLP)
DATASET 3: 20 NEWSGROUPS (20 CATEGORÍAS) | FRAMEWORK: KERAS
================================================================================
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import sklearn.metrics as metrics
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import keras
from keras import callbacks, layers, models, regularizers


# ==============================================================================
# SECCIÓN 1: CONFIGURACIÓN Y REPRODUCIBILIDAD
# ==============================================================================

def set_seed(seed: int = 42) -> None:
    """Fija semillas aleatorias para garantizar reproducibilidad."""
    random.seed(seed)
    np.random.seed(seed)
    keras.utils.set_random_seed(seed)


# ==============================================================================
# SECCIÓN 2: CARGA Y PREPROCESAMIENTO DE TEXTO (TF-IDF)
# ==============================================================================

def load_and_preprocess_data(
    vocab_size: int = 5000,
    val_split: float = 0.15,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, list[str], TfidfVectorizer]:
    """
    Carga el dataset 20 Newsgroups sin encabezados/firmas para evitar fugas de datos
    y transforma el texto crudo en vectores numéricos mediante TF-IDF.
    """
    print("\n--- SECCIÓN 2: CARGA Y VECTORIZACIÓN TF-IDF ---")
    
    # Remoción de metadatos (headers, footers, quotes) para evitar atajos espurios
    raw_train = fetch_20newsgroups(subset="train", remove=("headers", "footers", "quotes"), random_state=seed)
    raw_test = fetch_20newsgroups(subset="test", remove=("headers", "footers", "quotes"), random_state=seed)
    class_names = list(raw_train.target_names)

    # Partición estratificada: 85% entrenamiento y 15% validación
    train_texts, val_texts, train_labels, val_labels = train_test_split(
        raw_train.data, raw_train.target, test_size=val_split, random_state=seed, stratify=raw_train.target
    )

    # Vectorizador TF-IDF con filtrado de stop-words y frecuencia de términos sublineal
    vectorizer = TfidfVectorizer(
        max_features=vocab_size,
        stop_words="english",
        sublinear_tf=True,
        min_df=2,
        dtype=np.float32,
    )

    X_train = vectorizer.fit_transform(train_texts).toarray()
    X_val = vectorizer.transform(val_texts).toarray()
    X_test = vectorizer.transform(raw_test.data).toarray()

    y_train = np.array(train_labels, dtype=np.int32)
    y_val = np.array(val_labels, dtype=np.int32)
    y_test = np.array(raw_test.target, dtype=np.int32)

    print(f"Vocabulario: {vocab_size} términos | Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    return X_train, y_train, X_val, y_val, X_test, y_test, class_names, vectorizer


# ==============================================================================
# SECCIÓN 3: ARQUITECTURA DEL PERCEPTRÓN MULTICAPA (MLP)
# ==============================================================================

def build_mlp_model(
    input_dim: int,
    num_classes: int = 20,
    hidden_dim: int = 256,
    dropout_rate: float = 0.4,
    l2_reg: float = 1e-4,
    learning_rate: float = 1e-3,
) -> keras.Model:
    """
    Construye un MLP con 2 capas ocultas densas, activaciones ReLU, regularización L2,
    Dropout para controlar sobreajuste y capa de salida Softmax.
    """
    print("\n--- SECCIÓN 3: CONSTRUCCIÓN DEL MODELO MLP ---")

    reg = regularizers.l2(l2_reg) if l2_reg > 0 else None

    model = models.Sequential([
        layers.Input(shape=(input_dim,), name="tfidf_input"),
        
        # Capa Oculta 1: Extrae patrones de co-ocurrencia léxica
        layers.Dense(hidden_dim, kernel_regularizer=reg, activation="relu", name="dense_1"),
        layers.Dropout(dropout_rate, name="dropout_1"),

        # Capa Oculta 2: Comprime representaciones abstractas intermedias
        layers.Dense(hidden_dim // 2, kernel_regularizer=reg, activation="relu", name="dense_2"),
        layers.Dropout(dropout_rate, name="dropout_2"),

        # Capa de Salida: Distribución de probabilidad para las 20 clases
        layers.Dense(num_classes, activation="softmax", name="output_probabilities"),
    ], name="MLP_20Newsgroups")

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model


# ==============================================================================
# SECCIÓN 4: ENTRENAMIENTO Y CURVAS DE APRENDIZAJE
# ==============================================================================

def train_and_plot(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    output_prefix: str = "baseline_mlp",
) -> tuple[keras.callbacks.History, float, str]:
    """Entrena la red con EarlyStopping y guarda únicamente las curvas de pérdida y exactitud."""
    print("\n--- SECCIÓN 4: ENTRENAMIENTO DEL MODELO ---")
    os.makedirs("models", exist_ok=True)
    os.makedirs("results", exist_ok=True)

    cb_list = [
        callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2, verbose=1),
        callbacks.ModelCheckpoint(f"models/{output_prefix}.keras", monitor="val_loss", save_best_only=True),
    ]

    start_time = time.time()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cb_list,
        verbose=1,
    )
    total_time = time.time() - start_time
    print(f"Tiempo de entrenamiento: {total_time:.1f}s")

    # Gráfica única de Curvas de Pérdida y Aprendizaje (Train vs. Val)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(history.history["loss"], label="Train Loss", color="#1f77b4", lw=2)
    ax1.plot(history.history["val_loss"], label="Val Loss", color="#ff7f0e", lw=2)
    ax1.set_title("Curva de Pérdida (Cross-Entropy)")
    ax1.set_xlabel("Época")
    ax1.set_ylabel("Loss")
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend()

    ax2.plot(history.history["accuracy"], label="Train Accuracy", color="#2ca02c", lw=2)
    ax2.plot(history.history["val_accuracy"], label="Val Accuracy", color="#d62728", lw=2)
    ax2.set_title("Curva de Aprendizaje (Accuracy)")
    ax2.set_xlabel("Época")
    ax2.set_ylabel("Exactitud")
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend()

    plt.tight_layout()
    curves_path = f"results/learning_curves_{output_prefix}.png"
    plt.savefig(curves_path, dpi=200)
    plt.close()
    print(f"Curvas de pérdida y aprendizaje guardadas en: {curves_path}")

    return history, total_time, curves_path


# ==============================================================================
# SECCIÓN 5: EVALUACIÓN FORMAL DE MÉTRICAS
# ==============================================================================

def evaluate_model(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    class_names: list[str],
) -> dict:
    """Evalúa el modelo sobre el conjunto de test independiente y extrae métricas cuantitativas."""
    print("\n--- SECCIÓN 5: EVALUACIÓN EN CONJUNTO DE PRUEBA ---")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, batch_size=128, verbose=0), axis=1)

    macro_f1 = metrics.f1_score(y_test, y_pred, average="macro", zero_division=0)
    weighted_f1 = metrics.f1_score(y_test, y_pred, average="weighted", zero_division=0)
    precision = metrics.precision_score(y_test, y_pred, average="macro", zero_division=0)
    recall = metrics.recall_score(y_test, y_pred, average="macro", zero_division=0)

    print(f"Exactitud (Accuracy):  {test_acc:.4f} ({test_acc * 100:.2f}%)")
    print(f"F1-Score (Macro):      {macro_f1:.4f}")
    print(f"F1-Score (Ponderado):  {weighted_f1:.4f}")
    print(f"Precisión (Macro):     {precision:.4f}")
    print(f"Sensibilidad (Recall): {recall:.4f}")

    return {
        "test_loss": float(test_loss),
        "accuracy": float(test_acc),
        "f1_macro": float(macro_f1),
        "f1_weighted": float(weighted_f1),
        "precision_macro": float(precision),
        "recall_macro": float(recall),
    }


# ==============================================================================
# SECCIÓN 6: DEMO DE INFERENCIA Y FUNCIÓN PRINCIPAL (CLI)
# ==============================================================================

def run_quick_demo(model: keras.Model, vectorizer: TfidfVectorizer, class_names: list[str]) -> None:
    """
    Demostración de inferencia con casos claros (alta certeza) y casos mixtos/ambiguos
    (certeza promedio y distribución competitiva de probabilidades).
    """
    print("\n--- SECCIÓN 6: DEMO DE INFERENCIA (CASOS CLAROS Y MIXTOS) ---")
    samples = [
        # 1 y 2: Casos claros (alta confianza)
        ("NASA rover detects evidence of liquid water and atmosphere on Mars.", "Claro"),
        ("The engine needs a new oil filter and spark plugs for better mileage.", "Claro"),
        
        # 3, 4 y 5: Casos mixtos / frontera (confianza promedio con clases competidoras)
        ("A debate on whether doctors should perform experimental medical treatments when religious beliefs oppose it.", "Mixto (Medicina vs. Religión/Ética)"),
        ("I installed a new operating system on my PC to play baseball and hockey video games.", "Mixto (Sistemas vs. Deportes)"),
        ("Congressional debate on government funding for public healthcare hospitals and health insurance.", "Mixto (Política vs. Salud Pública)"),
    ]

    preds = model.predict(vectorizer.transform([s[0] for s in samples]).toarray(), verbose=0)
    for (text, tipo), prob in zip(samples, preds):
        top_idx = int(np.argmax(prob))
        top_indices = np.argsort(prob)[-3:][::-1]
        top3_str = ", ".join([f"{class_names[i]} ({prob[i]*100:.1f}%)" for i in top_indices])
        print(f"\n[{tipo}] \"{text}\"")
        print(f"  -> Predicción principal: {class_names[top_idx]} ({prob[top_idx]*100:.1f}%)")
        print(f"  -> Top 3 alternativas:   {top3_str}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Clasificador MLP en Keras para 20 Newsgroups (Entrega 1)")
    parser.add_argument("--vocab-size", type=int, default=5000, help="Tamaño del vocabulario TF-IDF")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Neuronas en primera capa oculta")
    parser.add_argument("--dropout", type=float, default=0.4, help="Tasa de Dropout (regularización)")
    parser.add_argument("--l2-reg", type=float, default=1e-4, help="Penalización L2 (Weight Decay)")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning Rate")
    parser.add_argument("--epochs", type=int, default=20, help="Número de épocas")
    parser.add_argument("--batch-size", type=int, default=64, help="Tamaño de lote")
    parser.add_argument("--output-name", type=str, default="baseline_mlp", help="Nombre del modelo y resultados")
    parser.add_argument("--quick-test", action="store_true", help="Prueba rápida de 2 épocas")
    args = parser.parse_args()

    if args.quick_test:
        args.epochs = 2
        args.vocab_size = 1500

    set_seed(42)

    # 1. Cargar y procesar datos
    X_train, y_train, X_val, y_val, X_test, y_test, class_names, vectorizer = load_and_preprocess_data(
        vocab_size=args.vocab_size
    )

    # Serializar vectorizador para inferencia futura
    with open(f"models/vectorizer_{args.output_name}.pkl", "wb") as f:
        pickle.dump(vectorizer, f)

    # 2. Construir arquitectura MLP
    model = build_mlp_model(
        input_dim=args.vocab_size,
        num_classes=len(class_names),
        hidden_dim=args.hidden_dim,
        dropout_rate=args.dropout,
        l2_reg=args.l2_reg,
        learning_rate=args.lr,
    )

    # 3. Entrenar y graficar curvas
    history, train_time, curves_path = train_and_plot(
        model, X_train, y_train, X_val, y_val,
        epochs=args.epochs, batch_size=args.batch_size, output_prefix=args.output_name
    )

    # 4. Evaluación en Test
    results = evaluate_model(model, X_test, y_test, class_names)
    results.update({
        "model_name": args.output_name,
        "vocab_size": args.vocab_size,
        "hidden_dim": args.hidden_dim,
        "dropout": args.dropout,
        "l2_reg": args.l2_reg,
        "lr": args.lr,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "training_time_sec": train_time,
        "curves_path": curves_path,
    })

    # Guardar métricas en JSON
    with open(f"results/{args.output_name}.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nMétricas consolidadas guardadas en: results/{args.output_name}.json")

    # 5. Demostración en vivo
    run_quick_demo(model, vectorizer, class_names)


if __name__ == "__main__":
    main()
