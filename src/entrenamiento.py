"""Fase 3 — Entrenamiento y evaluación de clasificadores clásicos.

Entrena Regresión Logística y SVM lineal sobre la representación TF-IDF del
corpus y reporta las métricas de la Fase 5 del anteproyecto:
Exactitud (Accuracy), Precisión, Sensibilidad (Recall) y F1-Score.

Pipeline completo:
    corpus_etiquetado.csv
        -> preprocesamiento (limpieza, tokens, stopwords, lematización)
        -> TF-IDF
        -> Regresión Logística / SVM
        -> métricas + matriz de confusión

Uso:
    python src/entrenamiento.py --input data/labeled/corpus_etiquetado.csv
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.svm import LinearSVC

try:  # ejecución como paquete: python -m src.entrenamiento
    from src.vectorizacion import construir_vectorizador
except ImportError:  # ejecución como script: python src/entrenamiento.py
    from vectorizacion import construir_vectorizador

try:
    try:
        from src.preprocessing import preprocesar
    except ImportError:
        from preprocessing import preprocesar
except Exception:  # spaCy puede no estar disponible al importar
    preprocesar = None


MODELOS = {
    "regresion_logistica": lambda: LogisticRegression(
        max_iter=1000, class_weight="balanced"
    ),
    "svm_lineal": lambda: LinearSVC(class_weight="balanced"),
}


def cargar_datos(ruta: str, columna_texto: str) -> tuple[list[str], pd.Series]:
    """Carga el CSV y devuelve (textos, etiquetas).

    Usa la columna `texto_procesado` si existe; si no, preprocesa `texto`
    al vuelo con el pipeline de spaCy.
    """
    df = pd.read_csv(ruta)
    if "texto_procesado" in df.columns:
        textos = df["texto_procesado"].astype(str).tolist()
    else:
        if preprocesar is None:
            raise ImportError(
                "No hay columna 'texto_procesado' y spaCy no está disponible. "
                "Ejecuta primero src/preprocessing.py o instala spaCy."
            )
        textos = [preprocesar(t) for t in df[columna_texto].astype(str)]
    return textos, df["sentimiento"]


def evaluar_holdout(textos, etiquetas, nombre_modelo, test_size, seed) -> None:
    """Entrena con un split train/test y reporta las métricas."""
    X_tr, X_te, y_tr, y_te = train_test_split(
        textos, etiquetas, test_size=test_size, random_state=seed,
        stratify=etiquetas,
    )
    vec = construir_vectorizador()
    X_tr_v = vec.fit_transform(X_tr)
    X_te_v = vec.transform(X_te)

    modelo = MODELOS[nombre_modelo]()
    modelo.fit(X_tr_v, y_tr)
    pred = modelo.predict(X_te_v)

    etiquetas_orden = ["negativo", "neutro", "positivo"]
    print(f"\n{'='*60}\nModelo: {nombre_modelo}  (hold-out, test={test_size:.0%})\n{'='*60}")
    print(f"Exactitud (Accuracy): {accuracy_score(y_te, pred):.4f}")
    print(f"F1-Score (macro):     {f1_score(y_te, pred, average='macro'):.4f}")
    print("\nReporte por clase (Precisión / Recall / F1):")
    print(classification_report(y_te, pred, labels=etiquetas_orden, zero_division=0))
    print("Matriz de confusión (filas=real, columnas=predicho):")
    cm = confusion_matrix(y_te, pred, labels=etiquetas_orden)
    print("            " + "  ".join(f"{e[:4]:>6}" for e in etiquetas_orden))
    for etiqueta, fila in zip(etiquetas_orden, cm):
        print(f"  {etiqueta:10s}" + "  ".join(f"{v:>6d}" for v in fila))


def evaluar_cv(textos, etiquetas, nombre_modelo, k, seed) -> None:
    """Validación cruzada estratificada (recomendada por el corpus pequeño)."""
    from sklearn.pipeline import Pipeline

    pipe = Pipeline([
        ("tfidf", construir_vectorizador()),
        ("clf", MODELOS[nombre_modelo]()),
    ])
    cv = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    pred = cross_val_predict(pipe, textos, etiquetas, cv=cv)

    etiquetas_orden = ["negativo", "neutro", "positivo"]
    print(f"\n{'='*60}\nModelo: {nombre_modelo}  ({k}-fold CV)\n{'='*60}")
    print(f"Exactitud (Accuracy): {accuracy_score(etiquetas, pred):.4f}")
    print(f"F1-Score (macro):     {f1_score(etiquetas, pred, average='macro'):.4f}")
    print("\nReporte por clase:")
    print(classification_report(etiquetas, pred, labels=etiquetas_orden, zero_division=0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Entrena y evalúa los clasificadores.")
    parser.add_argument("--input", default="data/labeled/corpus_etiquetado.csv")
    parser.add_argument("--columna", default="texto")
    parser.add_argument("--test-size", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv", type=int, default=0,
                        help="Si >0, usa validación cruzada con ese número de folds.")
    args = parser.parse_args()

    textos, etiquetas = cargar_datos(args.input, args.columna)
    print(f"Corpus: {len(textos)} noticias")
    print("Distribución:", etiquetas.value_counts().to_dict())

    for nombre in MODELOS:
        if args.cv > 0:
            evaluar_cv(textos, etiquetas, nombre, args.cv, args.seed)
        else:
            evaluar_holdout(textos, etiquetas, nombre, args.test_size, args.seed)


if __name__ == "__main__":
    main()
