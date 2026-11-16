"""Fase 5 — Evaluación integral y generación de gráficas.

Ejecuta el pipeline completo del proyecto y produce los resultados finales:

    1. Entrena Regresión Logística y SVM con validación cruzada y reporta las
       métricas de la Fase 5 (Accuracy, Precisión, Recall, F1) frente al corpus
       etiquetado (Gold Standard).
    2. Genera la matriz de confusión de cada modelo como imagen.
    3. Calcula el indicador S_t y lo grafica junto al precio del BTC.

Las gráficas se guardan en `resultados/`.

Uso:
    python src/evaluacion.py                    # usa precio si existe el CSV
    python src/evaluacion.py --cv 5
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # backend sin interfaz, para guardar PNG
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline

try:
    from src.vectorizacion import construir_vectorizador
    from src.entrenamiento import MODELOS, cargar_datos
    from src.indicador_sentimiento import calcular_indicador, cotejar_con_precio
except ImportError:
    from vectorizacion import construir_vectorizador
    from entrenamiento import MODELOS, cargar_datos
    from indicador_sentimiento import calcular_indicador, cotejar_con_precio

ETIQUETAS = ["negativo", "neutro", "positivo"]


def evaluar_modelos(textos, etiquetas, k, seed, dir_salida: Path) -> pd.DataFrame:
    """Evalúa todos los modelos con CV y devuelve una tabla resumen de métricas."""
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    resumen = []

    for nombre, factory in MODELOS.items():
        pipe = Pipeline([
            ("tfidf", construir_vectorizador()),
            ("clf", factory()),
        ])
        pred = cross_val_predict(pipe, textos, etiquetas, cv=cv)

        resumen.append({
            "modelo": nombre,
            "accuracy": accuracy_score(etiquetas, pred),
            "precision_macro": precision_score(etiquetas, pred, average="macro", zero_division=0),
            "recall_macro": recall_score(etiquetas, pred, average="macro", zero_division=0),
            "f1_macro": f1_score(etiquetas, pred, average="macro", zero_division=0),
        })

        print(f"\n{'='*60}\n{nombre}  ({k}-fold CV)\n{'='*60}")
        print(classification_report(etiquetas, pred, labels=ETIQUETAS, zero_division=0))

        # matriz de confusión como imagen
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            etiquetas, pred, labels=ETIQUETAS, ax=ax, cmap="Blues", colorbar=False
        )
        ax.set_title(f"Matriz de confusión — {nombre}")
        fig.tight_layout()
        ruta = dir_salida / f"matriz_confusion_{nombre}.png"
        fig.savefig(ruta, dpi=120)
        plt.close(fig)
        print(f"Matriz de confusión guardada -> {ruta}")

    return pd.DataFrame(resumen)


def graficar_sentimiento_precio(
    noticias_path: str, precio_path: str, frecuencia: str, dir_salida: Path
) -> None:
    """Grafica S_t frente al precio del BTC (doble eje Y)."""
    df_noticias = pd.read_csv(noticias_path)
    indicador = calcular_indicador(df_noticias, frecuencia)

    if not Path(precio_path).exists():
        print(f"\n(No se encontró {precio_path}; se omite la gráfica S_t vs precio. "
              "Ejecuta src/precio_btc.py para generarla.)")
        # igual graficamos S_t solo
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(indicador["periodo"], indicador["S_t"], "o-", color="tab:blue")
        ax.axhline(0, color="gray", lw=0.8, ls="--")
        ax.set_ylabel("Indicador S_t")
        ax.set_title("Indicador de sentimiento neto S_t")
        plt.xticks(rotation=45, ha="right")
        fig.tight_layout()
        ruta = dir_salida / "indicador_St.png"
        fig.savefig(ruta, dpi=120)
        plt.close(fig)
        print(f"Gráfica de S_t guardada -> {ruta}")
        return

    df_precio = pd.read_csv(precio_path)
    merged = cotejar_con_precio(indicador, df_precio)

    fig, ax1 = plt.subplots(figsize=(9, 4))
    ax1.plot(merged["periodo"], merged["S_t"], "o-", color="tab:blue", label="S_t")
    ax1.axhline(0, color="gray", lw=0.8, ls="--")
    ax1.set_ylabel("Indicador S_t", color="tab:blue")
    ax1.tick_params(axis="y", labelcolor="tab:blue")

    ax2 = ax1.twinx()
    ax2.plot(merged["periodo"], merged["precio_cierre"], "s-", color="tab:orange",
             label="Precio BTC")
    ax2.set_ylabel("Precio BTC (USD)", color="tab:orange")
    ax2.tick_params(axis="y", labelcolor="tab:orange")

    ax1.set_title("Sentimiento neto S_t vs precio del Bitcoin")
    plt.xticks(rotation=45, ha="right")
    fig.tight_layout()
    ruta = dir_salida / "sentimiento_vs_precio.png"
    fig.savefig(ruta, dpi=120)
    plt.close(fig)
    print(f"Gráfica S_t vs precio guardada -> {ruta}")

    validos = merged.dropna(subset=["S_t", "retorno_pct"])
    if len(validos) >= 3:
        corr = validos["S_t"].corr(validos["retorno_pct"])
        print(f"Correlación exploratoria S_t vs retorno diario: {corr:.3f}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluación integral (Fase 5).")
    parser.add_argument("--noticias", default="data/labeled/corpus_etiquetado.csv")
    parser.add_argument("--precio", default="data/raw/precio_btc.csv")
    parser.add_argument("--columna", default="texto")
    parser.add_argument("--frecuencia", choices=["D", "W"], default="D")
    parser.add_argument("--cv", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", default="resultados")
    args = parser.parse_args()

    dir_salida = Path(args.output)
    dir_salida.mkdir(parents=True, exist_ok=True)

    textos, etiquetas = cargar_datos(args.noticias, args.columna)
    print(f"Corpus: {len(textos)} noticias | distribución: {etiquetas.value_counts().to_dict()}")

    resumen = evaluar_modelos(textos, etiquetas, args.cv, args.seed, dir_salida)
    resumen_path = dir_salida / "resumen_metricas.csv"
    resumen.to_csv(resumen_path, index=False)
    print(f"\n{'='*60}\nResumen de métricas\n{'='*60}")
    print(resumen.to_string(index=False))
    print(f"\nResumen guardado -> {resumen_path}")

    graficar_sentimiento_precio(args.noticias, args.precio, args.frecuencia, dir_salida)
    print("\nEvaluación completa ✓")


if __name__ == "__main__":
    main()
