"""Fase 3 — Vectorización TF-IDF del corpus de noticias.

Convierte el texto preprocesado en una matriz numérica TF-IDF, tal como se
describe en el marco teórico del anteproyecto (sección 3.3):

    TF-IDF(t, d, D) = TF(t, d) x IDF(t, D)

El vectorizador de scikit-learn aplica exactamente este esquema (con la
normalización y el suavizado de IDF estándar de la librería).

Uso:
    from src.vectorizacion import construir_vectorizador, vectorizar

Uso como script (inspecciona el vocabulario del corpus):
    python src/vectorizacion.py --input data/processed/corpus_procesado.csv
"""
from __future__ import annotations

import argparse

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer


def construir_vectorizador(
    min_df: int = 2,
    max_df: float = 0.9,
    ngram_range: tuple[int, int] = (1, 2),
) -> TfidfVectorizer:
    """Crea el vectorizador TF-IDF con parámetros adecuados a un corpus pequeño.

    Parámetros
    ----------
    min_df : ignora términos que aparecen en menos de `min_df` documentos
        (elimina ruido / typos que aparecen una sola vez).
    max_df : ignora términos presentes en más del `max_df` de documentos
        (palabras demasiado comunes que no aportan discriminación).
    ngram_range : (1, 2) usa unigramas y bigramas, útil para capturar
        expresiones como "aversión riesgo" o "flujos positivos".
    """
    return TfidfVectorizer(
        min_df=min_df,
        max_df=max_df,
        ngram_range=ngram_range,
        sublinear_tf=True,  # 1 + log(tf), suaviza términos muy frecuentes
        norm="l2",
    )


def vectorizar(
    textos: pd.Series | list[str],
    vectorizador: TfidfVectorizer | None = None,
) -> tuple[csr_matrix, TfidfVectorizer]:
    """Ajusta el vectorizador y transforma los textos a una matriz TF-IDF."""
    if vectorizador is None:
        vectorizador = construir_vectorizador()
    matriz = vectorizador.fit_transform(textos)
    return matriz, vectorizador


def main() -> None:
    parser = argparse.ArgumentParser(description="Vectoriza el corpus con TF-IDF.")
    parser.add_argument("--input", default="data/processed/corpus_procesado.csv")
    parser.add_argument("--columna", default="texto_procesado")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    matriz, vec = vectorizar(df[args.columna].astype(str))

    print(f"Documentos: {matriz.shape[0]}")
    print(f"Términos en el vocabulario: {matriz.shape[1]}")
    print(f"Densidad de la matriz: {matriz.nnz / (matriz.shape[0] * matriz.shape[1]):.4%}")

    # términos con mayor peso IDF promedio (más discriminativos)
    import numpy as np

    pesos = np.asarray(matriz.mean(axis=0)).ravel()
    vocab = vec.get_feature_names_out()
    top = np.argsort(pesos)[::-1][:15]
    print("\nTop 15 términos por peso TF-IDF promedio:")
    for i in top:
        print(f"  {vocab[i]:30s} {pesos[i]:.4f}")


if __name__ == "__main__":
    main()
