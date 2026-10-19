"""
================================================================================
PROYECTO PRÁCTICO: REDES NEURONALES - UNIVERSIDAD DEL VALLE (2026-2)
ENTREGA 1: COMPARACIÓN DE RESULTADOS ENTRE VARIANTES DE MLP
================================================================================
Compara experimentos (por ejemplo, Baseline MLP vs. Tuned MLP con BatchNormalization/Dropout)
generando un reporte en Markdown y una gráfica comparativa de métricas.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_json_metrics(path: Path) -> dict | None:
    if not path.exists():
        print(f"Advertencia: Archivo no encontrado en {path.resolve()}")
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    parser = argparse.ArgumentParser(description="Comparador de Métricas de Experimentos MLP (Entrega 1)")
    parser.add_argument("--model-a", type=str, default="results/baseline_mlp.json", help="Ruta al JSON del modelo base")
    parser.add_argument("--model-b", type=str, default="results/tuned_mlp.json", help="Ruta al JSON del modelo comparado")
    parser.add_argument("--output-report", type=str, default="results/comparison_report.md", help="Ruta del reporte Markdown")
    parser.add_argument("--output-img", type=str, default="results/metrics_comparison.png", help="Ruta del gráfico comparativo")
    args = parser.parse_args()

    m_a = load_json_metrics(Path(args.model_a))
    m_b = load_json_metrics(Path(args.model_b))

    if m_a is None and m_b is None:
        print("Error: No se encontró ninguno de los archivos JSON de métricas especificados.")
        return

    name_a = m_a.get("model_name", "Modelo A") if m_a else "Modelo A"
    name_b = m_b.get("model_name", "Modelo B") if m_b else "Modelo B"

    print(f"\nComparando experimentos: '{name_a}' vs. '{name_b}'...\n")

    def get_val(data, keys, fmt=".4f"):
        if data is None:
            return "N/A"
        val = data
        for k in keys:
            if not isinstance(val, dict):
                return "N/A"
            val = val.get(k)
            if val is None:
                return "N/A"
        if isinstance(val, float):
            return f"{val:{fmt}}"
        return str(val)

    # Generación de la tabla Markdown
    table_lines = [
        "# Comparación de Desempeño: Experimentos MLP (Dataset 3: 20 Newsgroups)",
        "",
        "Este reporte técnico compara cuantitativamente el desempeño de diferentes configuraciones de red neuronal MLP.",
        "",
        f"| Métrica / Parámetro | {name_a} | {name_b} |",
        "| :--- | :---: | :---: |",
        f"| **Tamaño Vocabulario TF-IDF** | {get_val(m_a, ['dataset_info', 'vocab_size'])} | {get_val(m_b, ['dataset_info', 'vocab_size'])} |",
        f"| **Capas Ocultas** | {get_val(m_a, ['hyperparameters', 'num_layers'])} | {get_val(m_b, ['hyperparameters', 'num_layers'])} |",
        f"| **Neuronas (Capa 1)** | {get_val(m_a, ['hyperparameters', 'hidden_dim'])} | {get_val(m_b, ['hyperparameters', 'hidden_dim'])} |",
        f"| **Tasa Dropout** | {get_val(m_a, ['hyperparameters', 'dropout'])} | {get_val(m_b, ['hyperparameters', 'dropout'])} |",
        f"| **Batch Normalization** | {get_val(m_a, ['hyperparameters', 'use_batch_norm'])} | {get_val(m_b, ['hyperparameters', 'use_batch_norm'])} |",
        f"| **Learning Rate** | {get_val(m_a, ['hyperparameters', 'lr'])} | {get_val(m_b, ['hyperparameters', 'lr'])} |",
        "| --- | --- | --- |",
        f"| **Exactitud (Test Accuracy)** | {get_val(m_a, ['test_metrics', 'accuracy'])} | {get_val(m_b, ['test_metrics', 'accuracy'])} |",
        f"| **F1-Score (Macro)** | {get_val(m_a, ['test_metrics', 'f1_macro'])} | {get_val(m_b, ['test_metrics', 'f1_macro'])} |",
        f"| **F1-Score (Weighted)** | {get_val(m_a, ['test_metrics', 'f1_weighted'])} | {get_val(m_b, ['test_metrics', 'f1_weighted'])} |",
        f"| **Precisión (Macro)** | {get_val(m_a, ['test_metrics', 'precision_macro'])} | {get_val(m_b, ['test_metrics', 'precision_macro'])} |",
        f"| **Sensibilidad (Macro Recall)** | {get_val(m_a, ['test_metrics', 'recall_macro'])} | {get_val(m_b, ['test_metrics', 'recall_macro'])} |",
        f"| **Tiempo de Entrenamiento** | {get_val(m_a, ['training_time_sec'], '.2f')}s | {get_val(m_b, ['training_time_sec'], '.2f')}s |",
        f"| **Épocas Ejecutadas** | {get_val(m_a, ['hyperparameters', 'epochs_trained'])} | {get_val(m_b, ['hyperparameters', 'epochs_trained'])} |",
    ]

    report_content = "\n".join(table_lines)
    print(report_content)

    rep_path = Path(args.output_report)
    rep_path.parent.mkdir(parents=True, exist_ok=True)
    with open(rep_path, "w", encoding="utf-8") as f:
        f.write(report_content + "\n")
    print(f"\nReporte Markdown guardado en: {rep_path.resolve()}")

    # Generación de gráfica comparativa
    if m_a and m_b:
        metrics_names = ["Accuracy", "F1 (Macro)", "Precision (Macro)", "Recall (Macro)"]
        keys = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]

        vals_a = [float(m_a.get("test_metrics", {}).get(k, 0.0)) for k in keys]
        vals_b = [float(m_b.get("test_metrics", {}).get(k, 0.0)) for k in keys]

        x = np.arange(len(metrics_names))
        width = 0.35

        fig, ax = plt.subplots(figsize=(8, 5))
        rects1 = ax.bar(x - width/2, vals_a, width, label=name_a, color="#3498db")
        rects2 = ax.bar(x + width/2, vals_b, width, label=name_b, color="#2ecc71")

        ax.set_ylabel("Puntuación (0 a 1)")
        ax.set_title("Comparativa de Desempeño: Experimentos MLP")
        ax.set_xticks(x)
        ax.set_xticklabels(metrics_names)
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(axis="y", linestyle="--", alpha=0.5)

        for rect in rects1 + rects2:
            h = rect.get_height()
            ax.annotate(f"{h:.3f}",
                        xy=(rect.get_x() + rect.get_width() / 2, h),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=8)

        plt.tight_layout()
        img_path = Path(args.output_img)
        plt.savefig(img_path, dpi=200)
        plt.close()
        print(f"Gráfica comparativa guardada en: {img_path.resolve()}\n")


if __name__ == "__main__":
    main()
