# Clasificación de Texto en 20 Newsgroups: Perceptrón Multicapa (MLP) con Keras

Proyecto Práctico - Redes Neuronales (Universidad del Valle, 2026-2)  
**Entrega 1 (15%):** Clasificación multiclase de publicaciones de texto usando una arquitectura de **Perceptrón Multicapa (MLP)** en **Keras / TensorFlow**.

---

## 1. Definición del Proyecto y Objetivos

* **Conjunto de Datos:** **Dataset 3 (20 Newsgroups)**. Contiene aproximadamente 18.846 publicaciones distribuidas en 20 categorías temáticas organizadas en áreas generales como computación (`comp`), recreación y deportes (`rec`), ciencia (`sci`), política y religión (`talk`), entre otras.
* **Paradigma de Representación y Preprocesamiento:**
  * Eliminación de metadatos engañosos (`headers`, `footers`, `quotes`) para prevenir fuga de datos (*data leakage*) y evitar que el modelo aprenda atajos espurios (direcciones de correo, firmas u orígenes de red).
  * Vectorización con **TF-IDF** (`TfidfVectorizer`), empleando frecuencia de términos sublineal (`sublinear_tf=True`), eliminación de palabras vacías en inglés (`stop_words="english"`), umbral de frecuencia mínima (`min_df=2`) y vocabulario configurable (`--vocab-size`, por defecto 5.000 palabras).
  * Partición estratificada: 85% entrenamiento y 15% validación sobre el conjunto de entrenamiento, evaluando finalmente en el conjunto de prueba independiente (`subset="test"`).
* **Arquitectura de Red (MLP):**
  * Modelo secuencial implementado con `keras.models.Sequential`.
  * **Capa de Entrada:** Adaptada a la dimensión del vocabulario TF-IDF (`vocab_size`).
  * **Capa Oculta 1:** Capa densa con `hidden_dim` neuronas (por defecto 256), activación no lineal `ReLU` y penalización de regularización `L2` (Weight Decay).
  * **Regularización 1:** Capa de `Dropout` con tasa configurable (por defecto 0.4) para reducir co-adaptaciones complejas.
  * **Capa Oculta 2:** Capa densa con `hidden_dim // 2` neuronas (por defecto 128), activación no lineal `ReLU` y regularización `L2`.
  * **Regularización 2:** Capa de `Dropout` adicional (por defecto 0.4).
  * **Capa de Salida:** 20 neuronas con activación `Softmax` para producir la distribución de probabilidad sobre las 20 categorías, minimizando la pérdida `sparse_categorical_crossentropy`.
  * **Optimización y Callbacks:** Optimizador `Adam` asistido por:
    * `EarlyStopping`: Monitorea `val_loss` con paciencia de 4 épocas y restauración automática de los mejores pesos (`restore_best_weights=True`).
    * `ReduceLROnPlateau`: Reduce la tasa de aprendizaje a la mitad si la pérdida de validación se estanca durante 2 épocas consecutivas.
    * `ModelCheckpoint`: Guarda únicamente el mejor modelo en disco (`models/{output_name}.keras`).
* **Entregables:**
  1. **Código modular dividido en 6 secciones lógicas** en Python utilizando Keras (`train_mlp.py`).
  2. **Reporte técnico cuantitativo y visual** (Métricas JSON, curvas de pérdida y aprendizaje, y comparador de experimentos en Markdown y gráficos).
  3. **Presentación oral de 10 minutos con diapositivas y Demo Ejecutable interactiva** (`demo_inference.py`).

---

## 2. Estructura del Repositorio

Estructuración modular para reproducibilidad y sustentación del proyecto:

```text
RN_Entrega1/
├── train_mlp.py            # Script principal dividido en 6 secciones (datos, modelo, entreno, evaluación, demo)
├── demo_inference.py       # Interfaz de demo interactiva en vivo para la sustentación oral
├── compare_results.py      # Comparador de métricas y generador de reportes en Markdown y gráficos
├── README.md               # Documentación y guía de ejecución del proyecto
├── models/                 # Modelos guardados (.keras) y vectorizadores TF-IDF (.pkl)
│   ├── baseline_mlp.keras
│   └── vectorizer_baseline_mlp.pkl
└── results/                # Reportes JSON y curvas de aprendizaje (Loss y Accuracy)
    ├── baseline_mlp.json
    └── learning_curves_baseline_mlp.png
```

---

## 3. Requisitos e Instalación

* **Python:** 3.8 o superior (compatible con Python 3.10 / 3.11 / 3.12).
* **Instalación de dependencias:**
  ```bash
  pip install tensorflow keras scikit-learn matplotlib numpy
  ```

---

## 4. Parámetros del Script de Entrenamiento (`train_mlp.py`)

El script principal acepta los siguientes argumentos de línea de comandos para facilitar la experimentación:

| Argumento | Tipo | Por Defecto | Descripción |
| :--- | :---: | :---: | :--- |
| `--vocab-size` | `int` | `5000` | Tamaño máximo del vocabulario de palabras TF-IDF. |
| `--hidden-dim` | `int` | `256` | Neuronas en la primera capa oculta (la segunda capa oculta toma `hidden_dim // 2`). |
| `--dropout` | `float` | `0.4` | Tasa de regularización por Dropout aplicada tras cada capa oculta. |
| `--l2-reg` | `float` | `1e-4` | Coeficiente de penalización L2 (Weight Decay) en los pesos densos. |
| `--lr` | `float` | `1e-3` | Tasa de aprendizaje (Learning Rate) inicial del optimizador Adam. |
| `--epochs` | `int` | `20` | Número máximo de épocas de entrenamiento (sujeto a Early Stopping). |
| `--batch-size` | `int` | `64` | Tamaño del mini-lote (*mini-batch*). |
| `--output-name` | `str` | `baseline_mlp` | Nombre base para guardar los modelos, vectorizadores y resultados. |
| `--quick-test` | `flag` | `False` | Ejecuta un ciclo ultrarrápido (2 épocas, 1.500 palabras) para verificar el entorno. |

---

## 5. Guía de Ejecución Paso a Paso

### Paso 1: Verificación Rápida de Sanidad (Sanity Check)
Ejecuta 2 épocas sobre un subconjunto ligero para validar que el pipeline completo funciona sin errores:
```bash
python train_mlp.py --quick-test --output-name test_run
```

### Paso 2: Entrenamiento del Modelo MLP Baseline
Entrena el modelo base con el vocabulario estándar de 5.000 palabras y los hiperparámetros de referencia:
```bash
python train_mlp.py --vocab-size 5000 --hidden-dim 256 --dropout 0.4 --epochs 20 --output-name baseline_mlp
```
*(Nota: Dado que estos son los valores por defecto del script, también equivale a ejecutar simplemente `python train_mlp.py`).*

*Al finalizar generará automáticamente:*
* `models/baseline_mlp.keras`: Arquitectura y pesos del mejor punto de control (*checkpoint*).
* `models/vectorizer_baseline_mlp.pkl`: Vectorizador TF-IDF serializado para inferencias posteriores.
* `results/baseline_mlp.json`: Métricas de evaluación en el conjunto de prueba independiente (`test`).
* `results/learning_curves_baseline_mlp.png`: Gráfica combinada con las curvas de Pérdida (*Loss*) y Exactitud (*Accuracy*) para Train vs. Validación.

### Paso 3: Entrenamiento del Modelo MLP Optimizado / Tuned (Ajuste de Hiperparámetros)
Para explorar una configuración alternativa ajustando la capacidad de la red y los parámetros de regularización (por ejemplo, aumentando el vocabulario a 10.000 términos, primera capa oculta a 512 neuronas, menor Dropout y learning rate más fino):
```bash
python train_mlp.py --vocab-size 10000 --hidden-dim 512 --dropout 0.3 --l2-reg 0.00005 --lr 0.0005 --epochs 25 --output-name tuned_mlp
```
*Generará `models/tuned_mlp.keras`, `models/vectorizer_tuned_mlp.pkl`, `results/tuned_mlp.json` y `results/learning_curves_tuned_mlp.png`.*

### Paso 4: Comparación Cuantitativa y Generación de Reporte
Compara el modelo Baseline contra la variante ajustada:
```bash
python compare_results.py --model-a results/baseline_mlp.json --model-b results/tuned_mlp.json
```
*Generará automáticamente:*
* `results/comparison_report.md`: Tabla comparativa detallada en formato Markdown con hiperparámetros, métricas globales (Accuracy, F1 Macro, F1 Weighted, Precisión, Sensibilidad) y tiempos de cómputo.
* `results/metrics_comparison.png`: Gráfico de barras comparativo de métricas de desempeño.

### Paso 5: Demo Ejecutable en Vivo (Para la Sustentación Oral)
El script `demo_inference.py` soporta tres modalidades de uso:

1. **Modo Interactivo por Consola (Recomendado para la sustentación):**
   Permite escribir cualquier frase o noticia en inglés en tiempo real y muestra el top de categorías predichas con sus porcentajes y barras de confianza:
   ```bash
   python demo_inference.py --model-path models/baseline_mlp.keras --vectorizer-path models/vectorizer_baseline_mlp.pkl
   ```

2. **Modo Directo por Línea de Comandos:**
   Clasifica directamente una frase pasada como argumento:
   ```bash
   python demo_inference.py --text "NASA confirmed evidence of organic molecules and liquid water on Mars."
   ```

3. **Modo de Muestras Demostrativas Preconfiguradas:**
   Ejecuta automáticamente una batería de 5 casos de prueba diseñados para evaluar la certidumbre de la red:
   ```bash
   python demo_inference.py --run-samples
   ```

---

## 6. Casos de Prueba Demostrativos y Análisis de Comportamiento

Para la sustentación oral, la demo incluye dos tipologías de ejemplos:

* **Casos Claros (Alta Confianza y Dominio Específico):**
  * *"NASA rover detects evidence of liquid water and atmosphere on Mars."* $\rightarrow$ Predicción dominante en `sci.space` con alta certeza (>90%).
  * *"The engine needs a new oil filter and spark plugs for better mileage."* $\rightarrow$ Predicción dominante en `rec.autos` con alta certeza (>90%).
* **Casos Mixtos / Frontera (Intersección Temática y Distribución Competitiva):**
  * *"A debate on whether doctors should perform experimental medical treatments when religious beliefs oppose it."* $\rightarrow$ Frontera entre medicina (`sci.med`) y religión/ética (`talk.religion.misc`, `soc.religion.christian`).
  * *"I installed a new operating system on my PC to play baseball and hockey video games."* $\rightarrow$ Frontera entre sistemas computacionales (`comp.os.ms-windows.misc`, `comp.sys.ibm.pc.hardware`) y deportes (`rec.sport.baseball`, `rec.sport.hockey`).
  * *"Congressional debate on government funding for public healthcare hospitals and health insurance."* $\rightarrow$ Frontera entre política gubernamental (`talk.politics.misc`) y salud pública (`sci.med`).
