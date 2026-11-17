"""Inferencia y evaluación sobre noticias nuevas.

Uso:
    python src/predecir.py --nuevas data/raw/noticias_nuevas.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)
from sklearn.pipeline import Pipeline

try:
    from src.entrenamiento import MODELOS, cargar_datos
    from src.preprocessing import preprocesar
    from src.vectorizacion import construir_vectorizador
except ImportError:
    from entrenamiento import MODELOS, cargar_datos
    from preprocessing import preprocesar
    from vectorizacion import construir_vectorizador

ETIQUETAS = ["negativo", "neutro", "positivo"]
COLORES = {"positivo": "#2ecc71", "neutro": "#95a5a6", "negativo": "#e74c3c"}


def generar_graficas(df: pd.DataFrame, dir_salida: Path) -> None:
    """Genera resumen visual y matrices de confusión si existe etiqueta real."""
    # 1. Panel doble: Distribución por modelo + Confianza por noticia
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    conteos = pd.DataFrame({
        "Regresión Logística": df["pred_regresion_logistica"].value_counts(),
        "SVM Lineal": df["pred_svm_lineal"].value_counts(),
    }).reindex(ETIQUETAS).fillna(0)

    if "sentimiento" in df.columns:
        conteos.insert(0, "Real (Etiqueta)", df["sentimiento"].value_counts().reindex(ETIQUETAS).fillna(0))

    conteos.plot(kind="bar", ax=ax1, edgecolor="black")
    ax1.set_title("Distribución de sentimientos (Real vs Predicho)")
    ax1.set_ylabel("Cantidad de noticias")
    ax1.set_xlabel("")
    ax1.tick_params(axis="x", rotation=0)

    etiquetas_cortas = [
        (t[:42] + "...") if len(t) > 45 else t
        for t in (df["titulo"] if "titulo" in df.columns else df["texto"])
    ]
    colores_barras = [COLORES[s] for s in df["pred_regresion_logistica"]]

    y_pos = range(len(df))
    ax2.barh(y_pos, df["confianza_lr"], color=colores_barras, edgecolor="black")
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(etiquetas_cortas, fontsize=8)
    ax2.invert_yaxis()
    ax2.set_xlim(0, 1.05)
    ax2.set_xlabel("Confianza (Probabilidad máxima LR)")
    ax2.set_title("Predicción y confianza por noticia")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=COLORES[e], label=e.capitalize())
        for e in ETIQUETAS
    ]
    ax2.legend(handles=handles, loc="lower right", title="Predicción LR")

    fig.tight_layout()
    ruta_panel = dir_salida / "predicciones_nuevas_resumen.png"
    fig.savefig(ruta_panel, dpi=150)
    plt.close(fig)
    print(f"Gráfica resumen guardada -> {ruta_panel}")

    # 2. Si hay columna 'sentimiento', generar matrices de confusión reales
    if "sentimiento" in df.columns:
        for modelo_col, sufijo, titulo in [
            ("pred_regresion_logistica", "lr", "Regresión Logística"),
            ("pred_svm_lineal", "svm", "SVM Lineal"),
        ]:
            fig, ax = plt.subplots(figsize=(5, 4))
            ConfusionMatrixDisplay.from_predictions(
                df["sentimiento"],
                df[modelo_col],
                labels=ETIQUETAS,
                ax=ax,
                cmap="Blues",
                colorbar=False,
            )
            ax.set_title(f"Matriz de confusión (Nuevas) — {titulo}")
            fig.tight_layout()
            ruta_cm = dir_salida / f"matriz_confusion_nuevas_{sufijo}.png"
            fig.savefig(ruta_cm, dpi=150)
            plt.close(fig)
            print(f"Matriz de confusión real guardada -> {ruta_cm}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Predice y evalúa noticias nuevas.")
    parser.add_argument("--train", default="data/processed/corpus_procesado.csv")
    parser.add_argument("--nuevas", default="data/raw/noticias_nuevas.csv")
    parser.add_argument("--output", default="resultados/predicciones_nuevas.csv")
    args = parser.parse_args()

    ruta_train = args.train if Path(args.train).exists() else "data/labeled/corpus_etiquetado.csv"
    X_train, y_train = cargar_datos(ruta_train, columna_texto="texto")

    pipelines = {}
    for nombre, factory in MODELOS.items():
        pipe = Pipeline([
            ("tfidf", construir_vectorizador()),
            ("clf", factory()),
        ])
        pipe.fit(X_train, y_train)
        pipelines[nombre] = pipe

    df_nuevas = pd.read_csv(args.nuevas)
    if "texto" not in df_nuevas.columns:
        if {"titulo", "resumen"}.issubset(df_nuevas.columns):
            df_nuevas["texto"] = (
                df_nuevas["titulo"].fillna("") + ". " + df_nuevas["resumen"].fillna("")
            )
        else:
            raise ValueError("El CSV debe tener 'texto' o 'titulo' y 'resumen'.")

    textos_nuevos_proc = [preprocesar(t) for t in df_nuevas["texto"].astype(str)]

    for nombre, pipe in pipelines.items():
        df_nuevas[f"pred_{nombre}"] = pipe.predict(textos_nuevos_proc)

    probs_lr = pipelines["regresion_logistica"].predict_proba(textos_nuevos_proc)
    df_nuevas["confianza_lr"] = probs_lr.max(axis=1).round(3)

    # Si tiene etiqueta real, marcar aciertos y mostrar reporte de evaluación
    if "sentimiento" in df_nuevas.columns:
        df_nuevas["acierto_lr"] = df_nuevas["sentimiento"] == df_nuevas["pred_regresion_logistica"]
        df_nuevas["acierto_svm"] = df_nuevas["sentimiento"] == df_nuevas["pred_svm_lineal"]

        for nombre in MODELOS:
            pred = df_nuevas[f"pred_{nombre}"]
            real = df_nuevas["sentimiento"]
            print(f"\n{'='*60}\nEVALUACIÓN EN NOTICIAS NUEVAS — {nombre}\n{'='*60}")
            print(f"Exactitud (Accuracy): {accuracy_score(real, pred):.4f}")
            print(f"F1-Score (macro):     {f1_score(real, pred, average='macro', zero_division=0):.4f}")
            print(classification_report(real, pred, labels=ETIQUETAS, zero_division=0))

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    df_nuevas.to_csv(salida, index=False)

    generar_graficas(df_nuevas, salida.parent)

    cols_base = ["titulo", "sentimiento", "pred_regresion_logistica", "confianza_lr", "pred_svm_lineal"]
    cols_mostrar = [c for c in cols_base if c in df_nuevas.columns]
    print("\n" + "=" * 75)
    print(df_nuevas[cols_mostrar].to_string(index=False))
    print(f"\nResultados guardados en -> {salida}")


if __name__ == "__main__":
    main()