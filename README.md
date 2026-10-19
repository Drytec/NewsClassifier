# Clasificación de Texto en 20 Newsgroups: Perceptrón Multicapa (MLP) con Keras

Proyecto Práctico - Redes Neuronales (Universidad del Valle, 2026-2)  
**Entrega 1 (15%):** Clasificación multiclase de publicaciones de texto usando una arquitectura de **Perceptrón Multicapa (MLP)** en **Keras / TensorFlow**.

---

## 1. Definición del Proyecto y Objetivos

* **Conjunto de Datos:** **Dataset 3 (20 Newsgroups)**. Contiene aproximadamente 20.000 mensajes distribuidos en 20 categorías temáticas organizadas en áreas como computación, deportes, ciencia, política y religión.
* **Paradigma de Representación:** Vectorización de texto con **TF-IDF** (Sublinear Term Frequency, eliminación de stop-words y vocabulario configurable) con prevención de fuga de datos (limpieza de encabezados/firmas que contienen metadatos engañosos).
* **Arquitectura de Red:** **Perceptrón Multicapa (MLP)** implementado con `keras.models.Sequential`, incorporando:
  * Capa de entrada adaptada a la dimensión del vocabulario TF-IDF.
  * Capas densas intermedias con funciones de activación no lineales (`ReLU`).
  * Mecanismos de regularización y estabilización: `Dropout`, `BatchNormalization` y regularización `L2` (Weight Decay).
  * Capa de salida con activación `Softmax` (20 neuronas) y función de coste `sparse_categorical_crossentropy`.
  * Optimización mediante `Adam` con callbacks adaptativos (`EarlyStopping`, `ReduceLROnPlateau`, `ModelCheckpoint`).
* **Entregables:**
  1. **Código modular dividido en secciones** en Python utilizando Keras.
  2. **Reporte técnico estructurado** (Resumen, Introducción, Descripción del dataset, Arquitectura MLP, Ajuste de parámetros, Análisis de resultados, Conclusiones y Bibliografía).
  3. **Presentación oral de 10 minutos con diapositivas y Demo Ejecutable interactiva**.

---

## 2. Estructura del Repositorio

Inspirada en las mejores prácticas de estructuración de proyectos de Deep Learning:

```text
RN_Entrega1/
├── train_mlp.py            # Script principal dividido en 8 secciones detalladas
├── demo_inference.py       # Interfaz de demo ejecutable en vivo para la sustentación
├── compare_results.py      # Comparador de métricas y generador de reportes en Markdown
├── README.md               # Documentación y guía de ejecución
├── models/                 # Modelos guardados (.keras) y vectorizadores TF-IDF (.pkl)
│   ├── baseline_mlp.keras
│   └── vectorizer_baseline_mlp.pkl
└── results/                # Reportes JSON, matrices de confusión y curvas de aprendizaje
    ├── baseline_mlp.json
    ├── learning_curves_baseline_mlp.png
    └── confusion_matrix_baseline_mlp.png
```

---

## 3. Requisitos e Instalación

* **Python:** 3.8 o superior (compatible con Python 3.10 / 3.11 / 3.12).
* **Dependencias:**
  ```bash
  pip install tensorflow keras scikit-learn matplotlib numpy
  ```

---

## 4. Guía de Ejecución Paso a Paso

### Paso 1: Verificación Rápida (Sanity Check)
Ejecuta 2 épocas sobre un subconjunto ligero para comprobar que todo el pipeline funciona:
```bash
python train_mlp.py --quick-test --output-name test_run
```

### Paso 2: Entrenamiento del Modelo MLP Baseline
Entrena el modelo base con el vocabulario completo (10.000 palabras) y parámetros estándar:
```bash
python train_mlp.py --vocab-size 10000 --hidden-dim 256 --num-layers 2 --dropout 0.3 --epochs 25 --output-name baseline_mlp
```
*Generará automáticamente:*
* `models/baseline_mlp.keras` (Pesos del mejor modelo por validación).
* `models/vectorizer_baseline_mlp.pkl` (Vocabulario TF-IDF serializado).
* `results/baseline_mlp.json` (Métricas globales y por clase).
* `results/learning_curves_baseline_mlp.png` (Curvas de Loss y Accuracy de Train vs. Val).
* `results/confusion_matrix_baseline_mlp.png` (Matriz de confusión detallada).

### Paso 3: Entrenamiento del Modelo MLP Optimizado / Tuned (Elemento adicional para rúbrica)
Entrena una segunda variante ajustando hiperparámetros (mayor capacidad, n-gramas bigramas o regularización):
```bash
python train_mlp.py --vocab-size 15000 --ngram-max 2 --hidden-dim 512 --num-layers 3 --dropout 0.4 --epochs 30 --output-name tuned_mlp
```

### Paso 4: Comparación Cuantitativa y Generación de Reporte
Compara el modelo Baseline contra el modelo Optimizado:
```bash
python compare_results.py --model-a results/baseline_mlp.json --model-b results/tuned_mlp.json
```
*Esto generará `results/comparison_report.md` con la tabla comparativa y `results/metrics_comparison.png`.*

### Paso 5: Demo Ejecutable en Vivo (Para la Sustentación Oral)
Para realizar la demostración en tiempo real durante la presentación de 10 minutos:
```bash
python demo_inference.py --model-path models/baseline_mlp.keras --vectorizer-path models/vectorizer_baseline_mlp.pkl
```
*Permite escribir cualquier frase o noticia en inglés y muestra las categorías predichas con sus porcentajes de confianza en forma de barras gráficas en la terminal.*
