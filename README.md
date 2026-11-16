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
│   └── processed/        # texto ya preprocesado (Fase 3)
├── src/
│   ├── preprocessing.py  # limpieza, tokens, stopwords, lematización
│   └── build_dataset.py  # genera el CSV de etiquetado
├── requirements.txt
└── README.md
```

## Fases del proyecto

- **Fase 1** — Construcción del corpus (180 noticias)
- **Fase 2** — Etiquetado y setup ← estás aquí
- **Fase 3** — Vectorización TF-IDF + entrenamiento
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

> El etiquetado es una **propuesta base** generada automáticamente siguiendo un
> criterio financiero. El equipo debe revisarlo manualmente para construir el
> *Gold Standard* (ver Fase 2 del anteproyecto).
