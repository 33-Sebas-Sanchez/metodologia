"""Preprocesamiento de texto para noticias financieras en español.

Pipeline (según el marco teórico del anteproyecto):
    1. Limpieza: minúsculas, quita HTML, URLs, números y signos.
    2. Tokenización.
    3. Eliminación de stopwords.
    4. Lematización.

Uso como script:
    python src/preprocessing.py \
        --input data/labeled/corpus_etiquetado.csv \
        --output data/processed/corpus_procesado.csv
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

try:
    import spacy
except ImportError:  # pragma: no cover
    spacy = None

# Expresiones regulares de limpieza
RE_HTML = re.compile(r"<[^>]+>")
RE_URL = re.compile(r"http\S+|www\.\S+")
RE_NO_ALFA = re.compile(r"[^a-záéíóúñü\s]")
RE_ESPACIOS = re.compile(r"\s+")

_NLP = None  # modelo spaCy cargado de forma perezosa


def cargar_modelo():
    """Carga el modelo de spaCy en español una sola vez."""
    global _NLP
    if _NLP is None:
        if spacy is None:
            raise ImportError(
                "spaCy no está instalado. Ejecuta: pip install spacy && "
                "python -m spacy download es_core_news_sm"
            )
        try:
            _NLP = spacy.load("es_core_news_sm", disable=["ner", "parser"])
        except OSError as exc:  # pragma: no cover
            raise OSError(
                "Falta el modelo 'es_core_news_sm'. Ejecuta: "
                "python -m spacy download es_core_news_sm"
            ) from exc
    return _NLP


def limpiar(texto: str) -> str:
    """Paso 1: normaliza y elimina ruido del texto."""
    texto = texto.lower()
    texto = RE_HTML.sub(" ", texto)
    texto = RE_URL.sub(" ", texto)
    # conserva tildes y ñ (importantes en español); elimina el resto
    texto = RE_NO_ALFA.sub(" ", texto)
    texto = RE_ESPACIOS.sub(" ", texto).strip()
    return texto


def preprocesar(texto: str) -> str:
    """Aplica el pipeline completo y devuelve texto lematizado sin stopwords."""
    nlp = cargar_modelo()
    limpio = limpiar(texto)
    doc = nlp(limpio)
    lemas = [
        token.lemma_
        for token in doc
        if not token.is_stop and not token.is_punct and len(token.lemma_) > 2
    ]
    return " ".join(lemas)


def procesar_dataframe(df: pd.DataFrame, columna: str = "texto") -> pd.DataFrame:
    """Agrega la columna 'texto_procesado' al DataFrame."""
    df = df.copy()
    df["texto_procesado"] = df[columna].astype(str).map(preprocesar)
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Preprocesa el corpus de noticias.")
    parser.add_argument("--input", default="data/labeled/corpus_etiquetado.csv")
    parser.add_argument("--output", default="data/processed/corpus_procesado.csv")
    parser.add_argument("--columna", default="texto")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df = procesar_dataframe(df, columna=args.columna)

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida, index=False)
    print(f"Procesadas {len(df)} noticias -> {salida}")
    print("\nEjemplo:")
    print(df[["texto", "texto_procesado"]].head(1).to_string(index=False))


if __name__ == "__main__":
    main()
