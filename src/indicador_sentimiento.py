"""Fase 4 — Indicador de sentimiento neto S_t y cotejo con el precio de BTC.

Calcula el indicador sintético de sentimiento neto definido en el marco teórico
del anteproyecto (sección 3.6):

    S_t = (N_pos,t - N_neg,t) / (N_pos,t + N_neg,t + N_neu,t)

donde N_pos,t, N_neg,t y N_neu,t son el número de noticias clasificadas como
positivas, negativas y neutras en el periodo t (diario o semanal).

Luego lo cruza con el histórico de precios y reporta la correlación exploratoria
entre el sentimiento y la variación del precio (Objetivo Específico 3).

Uso:
    python src/indicador_sentimiento.py \
        --noticias data/labeled/corpus_etiquetado.csv \
        --precio data/raw/precio_btc.csv \
        --frecuencia D
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def calcular_indicador(df_noticias: pd.DataFrame, frecuencia: str = "D") -> pd.DataFrame:
    """Calcula S_t por periodo (D=diario, W=semanal).

    Devuelve un DataFrame con: periodo, n_pos, n_neg, n_neu, total, S_t.
    """
    df = df_noticias.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])

    # conteo de cada clase por periodo
    conteo = (
        df.groupby([pd.Grouper(key="fecha", freq=frecuencia), "sentimiento"])
        .size()
        .unstack(fill_value=0)
    )
    for col in ("positivo", "negativo", "neutro"):
        if col not in conteo:
            conteo[col] = 0

    conteo = conteo.rename(
        columns={"positivo": "n_pos", "negativo": "n_neg", "neutro": "n_neu"}
    )
    conteo["total"] = conteo["n_pos"] + conteo["n_neg"] + conteo["n_neu"]

    # S_t = (n_pos - n_neg) / total ; evita división por cero
    conteo["S_t"] = (conteo["n_pos"] - conteo["n_neg"]) / conteo["total"].where(
        conteo["total"] > 0, other=pd.NA
    )

    conteo = conteo.reset_index().rename(columns={"fecha": "periodo"})
    conteo["periodo"] = conteo["periodo"].dt.strftime("%Y-%m-%d")
    return conteo[["periodo", "n_pos", "n_neg", "n_neu", "total", "S_t"]]


def cotejar_con_precio(
    indicador: pd.DataFrame, df_precio: pd.DataFrame
) -> pd.DataFrame:
    """Une S_t con el precio y calcula el retorno diario del BTC."""
    precio = df_precio.copy()
    precio["fecha"] = pd.to_datetime(precio["fecha"]).dt.strftime("%Y-%m-%d")
    precio = precio.rename(columns={"fecha": "periodo"})

    merged = indicador.merge(precio, on="periodo", how="inner")
    merged = merged.sort_values("periodo").reset_index(drop=True)
    merged["retorno_pct"] = merged["precio_cierre"].pct_change() * 100
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Calcula S_t y lo cruza con el precio.")
    parser.add_argument("--noticias", default="data/labeled/corpus_etiquetado.csv")
    parser.add_argument("--precio", default="data/raw/precio_btc.csv")
    parser.add_argument("--frecuencia", choices=["D", "W"], default="D")
    parser.add_argument("--output", default="data/processed/indicador_sentimiento.csv")
    args = parser.parse_args()

    df_noticias = pd.read_csv(args.noticias)
    indicador = calcular_indicador(df_noticias, args.frecuencia)

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    indicador.to_csv(salida, index=False)
    print(f"Indicador S_t calculado ({args.frecuencia}) -> {salida}")
    print(indicador.to_string(index=False))

    # cotejo con el precio si existe el archivo
    precio_path = Path(args.precio)
    if precio_path.exists():
        df_precio = pd.read_csv(precio_path)
        merged = cotejar_con_precio(indicador, df_precio)
        salida_merge = salida.with_name("sentimiento_vs_precio.csv")
        merged.to_csv(salida_merge, index=False)
        print(f"\nCotejo sentimiento vs precio -> {salida_merge}")
        print(merged.to_string(index=False))

        validos = merged.dropna(subset=["S_t", "retorno_pct"])
        if len(validos) >= 3:
            corr = validos["S_t"].corr(validos["retorno_pct"])
            print(f"\nCorrelación exploratoria S_t vs retorno diario: {corr:.3f}")
            print("(Análisis exploratorio; no implica causalidad — Objetivo Específico 3.)")
        else:
            print("\nFaltan periodos coincidentes para estimar la correlación.")
    else:
        print(f"\n(No se encontró {precio_path}. Ejecuta primero src/precio_btc.py "
              "para el cotejo con el precio.)")


if __name__ == "__main__":
    main()
