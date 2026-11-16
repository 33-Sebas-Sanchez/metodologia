# Análisis de Sentimiento en Noticias Financieras sobre Bitcoin (Español)

Proyecto de NLP clásico (TF-IDF + Regresión Logística / SVM) para clasificar la
polaridad de noticias financieras en español y explorar su relación con el precio
del Bitcoin.

## Estructura

```
sentimiento-btc/
├── data/
│   ├── raw/              # corpus original
│   ├── labeled/          # corpus_etiquetado.csv (180 noticias con sentimiento)
│   └── processed/        # texto ya preprocesado
├── src/
│   ├── preprocessing.py     # limpieza, tokens, stopwords, lematización
│   ├── resumen_etiquetado.py # control de calidad del etiquetado
│   ├── vectorizacion.py     # TF-IDF (Fase 3)
│   └── entrenamiento.py     # Regresión Logística + SVM + métricas (Fase 3)
├── requerimientos.txt
└── README.md
```

## Fases del proyecto

- **Fase 1** — Construcción del corpus (180 noticias)
- **Fase 2** — Etiquetado y setup
- **Fase 3** — Vectorización TF-IDF + entrenamiento ← estás aquí
- **Fase 4** — Datos de precio BTC + indicador S_t
- **Fase 5** — Evaluación (Accuracy, Precision, Recall, F1)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
python -m spacy download es_core_news_sm
```

## Columnas de `data/labeled/corpus_etiquetado.csv`

| Columna      | Descripción                                               |
|--------------|-----------------------------------------------------------|
| `id`         | Identificador de la noticia (1–180)                       |
| `titulo`     | Titular de la noticia                                     |
| `resumen`    | Mini resumen                                              |
| `texto`      | `titulo + ". " + resumen` (insumo del modelo)             |
| `fuente`     | Medio de publicación                                      |
| `fecha`      | Fecha de publicación                                      |
| `categoria`  | Eje temático del corpus (1–6)                             |
| `sentimiento`| Etiqueta: `positivo`, `negativo` o `neutro`               |



## Ejecución de la Fase 3 (vectorización + entrenamiento)

Desde la raíz del proyecto y con el entorno instalado:

```bash
# 1. Preprocesar el corpus (genera data/processed/corpus_procesado.csv)
python src/preprocessing.py

# 2. (Opcional) Inspeccionar el vocabulario TF-IDF
python src/vectorizacion.py

# 3. Entrenar y evaluar los dos clasificadores
python src/entrenamiento.py                 # split hold-out 75/25
python src/entrenamiento.py --cv 5           # validación cruzada 5-fold (recomendado)
```

`entrenamiento.py` reporta, para Regresión Logística y SVM lineal, las métricas de
la Fase 5: **Exactitud (Accuracy), Precisión, Sensibilidad (Recall) y F1-Score**,
más la matriz de confusión.

### Decisiones de diseño

- **TF-IDF** con unigramas + bigramas, `min_df=2`, `max_df=0.9` y `sublinear_tf`
  (ver `src/vectorizacion.py`). Esto corresponde al esquema TF x IDF del marco teórico.
- **`class_weight="balanced"`** en ambos modelos, porque el corpus está desbalanceado
  (50% positivo, 27% neutro, 23% negativo).
- Se recomienda **validación cruzada** (`--cv 5`) frente al hold-out por el tamaño
  acotado del corpus (150–250 noticias), reduciendo la varianza de la estimación.

### Nota metodológica (importante para el informe)

El corpus contiene muchas noticias casi duplicadas (cada nota base se repite como
"Informe #2…#7"). Esto, sumado a un vocabulario muy marcado por clase, hace que las
métricas salgan **muy altas** (posible sobreestimación). Conviene documentarlo en
"Limitaciones" y, si se quiere una estimación más realista, agrupar los duplicados
antes de separar train/test para evitar fuga de información entre particiones.
