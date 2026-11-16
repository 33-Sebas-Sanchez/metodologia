"""Valida y resume el CSV etiquetado (control de calidad de la Fase 2)."""
from __future__ import annotations

import argparse

import pandas as pd

ETIQUETAS_VALIDAS = {"positivo", "negativo", "neutro"}
COLUMNAS_ESPERADAS = {
    "id", "titulo", "resumen", "texto", "fuente", "fecha", "categoria", "sentimiento",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/labeled/corpus_etiquetado.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)

    faltantes = COLUMNAS_ESPERADAS - set(df.columns)
    assert not faltantes, f"Faltan columnas: {faltantes}"

    invalidas = set(df["sentimiento"].unique()) - ETIQUETAS_VALIDAS
    assert not invalidas, f"Etiquetas inválidas: {invalidas}"
    assert df["id"].is_unique, "Hay ids duplicados"
    assert df["texto"].notna().all(), "Hay textos vacíos"

    print(f"Total de noticias: {len(df)}")
    print("\nDistribución de sentimiento:")
    print(df["sentimiento"].value_counts().to_string())
    print("\nSentimiento por categoría temática:")
    print(pd.crosstab(df["categoria"], df["sentimiento"]).to_string())
    print("\nValidación OK ✓")


if __name__ == "__main__":
    main()
