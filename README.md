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
│   ├── entrenamiento.py     # Regresión Logística + SVM + métricas (Fase 3)
│   ├── precio_btc.py        # descarga histórico de precio BTC (Fase 4)
│   ├── indicador_sentimiento.py # indicador S_t + cotejo con precio (Fase 4)
│   └── evaluacion.py        # evaluación integral + gráficas (Fase 5)
├── resultados/              # métricas y gráficas generadas (Fase 5)
├── requerimientos.txt
└── README.md
```

## Fases del proyecto

- **Fase 1** — Construcción del corpus (180 noticias)
- **Fase 2** — Etiquetado y setup
- **Fase 3** — Vectorización TF-IDF + entrenamiento
- **Fase 4** — Datos de precio BTC + indicador S_t
- **Fase 5** — Evaluación (Accuracy, Precision, Recall, F1) ← estás aquí

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

## Ejecución de la Fase 4 (precio BTC + indicador S_t)

```bash
# 1. Descargar el histórico de precios de BTC (elige una fuente)
python src/precio_btc.py --fuente coingecko --dias 60
#   o, si prefieres yfinance:
# python src/precio_btc.py --fuente yfinance --inicio 2026-09-01 --fin 2026-10-20

# 2. Calcular el indicador S_t y cruzarlo con el precio
python src/indicador_sentimiento.py --frecuencia D    # diario
python src/indicador_sentimiento.py --frecuencia W    # semanal
```

### Indicador de sentimiento neto

Según la sección 3.6 del anteproyecto:

```
S_t = (N_pos,t - N_neg,t) / (N_pos,t + N_neg,t + N_neu,t)
```

`indicador_sentimiento.py` genera:
- `data/processed/indicador_sentimiento.csv` — S_t por periodo.
- `data/processed/sentimiento_vs_precio.csv` — S_t cruzado con el precio y el retorno diario.
- La **correlación exploratoria** entre S_t y el retorno del BTC (Objetivo Específico 3).

> El análisis es **exploratorio**: una correlación no implica causalidad, y el corpus
> cubre un periodo corto (oct. 2026). Interpretar con cautela en el informe.

## Ejecución de la Fase 5 (evaluación integral)

`evaluacion.py` es el orquestador final: entrena los dos modelos con validación
cruzada, reporta todas las métricas frente al Gold Standard y genera las gráficas.

```bash
# (recomendado) descargar primero el precio para la gráfica S_t vs precio
python src/precio_btc.py --fuente coingecko --dias 60

# evaluación completa (métricas + matrices de confusión + gráfica S_t vs precio)
python src/evaluacion.py --cv 5
```

Genera en `resultados/`:
- `resumen_metricas.csv` — Accuracy, Precisión, Recall y F1 (macro) de cada modelo.
- `matriz_confusion_regresion_logistica.png` y `matriz_confusion_svm_lineal.png`.
- `sentimiento_vs_precio.png` — S_t y precio del BTC en doble eje (o `indicador_St.png`
  si no hay datos de precio).

### Pipeline completo de principio a fin

```bash
python src/preprocessing.py            # Fase 2/3: preprocesa el corpus
python src/precio_btc.py --dias 60     # Fase 4: precio de BTC
python src/evaluacion.py --cv 5        # Fase 3/5: entrena, evalúa y grafica
```
