"""Fase 4 — Descarga del histórico de precios del Bitcoin.

Obtiene el precio de cierre diario de BTC en USD y lo guarda como CSV con
columnas `fecha,precio_cierre` para cotejarlo luego con el indicador de
sentimiento neto S_t.

Fuentes soportadas:
    - yfinance  (requiere `pip install yfinance`)
    - CoinGecko (API pública, sin API key; requiere `requests`)

Uso:
    python src/precio_btc.py --fuente coingecko --dias 60
    python src/precio_btc.py --fuente yfinance --inicio 2026-09-01 --fin 2026-10-20
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def descargar_yfinance(inicio: str, fin: str) -> pd.DataFrame:
    """Descarga OHLC de BTC-USD con yfinance y devuelve fecha + cierre."""
    import yfinance as yf

    df = yf.download("BTC-USD", start=inicio, end=fin, progress=False)
    if df.empty:
        raise RuntimeError("yfinance no devolvió datos. Revisa el rango de fechas.")
    df = df.reset_index()[["Date", "Close"]]
    df.columns = ["fecha", "precio_cierre"]
    df["fecha"] = pd.to_datetime(df["fecha"]).dt.strftime("%Y-%m-%d")
    return df


def descargar_coingecko(dias: int) -> pd.DataFrame:
    """Descarga el precio diario de BTC de los últimos `dias` vía CoinGecko."""
    import requests

    url = "https://api.coingecko.com/api/v3/coins/bitcoin/market_chart"
    params = {"vs_currency": "usd", "days": dias, "interval": "daily"}
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    precios = resp.json()["prices"]  # [[timestamp_ms, precio], ...]

    df = pd.DataFrame(precios, columns=["timestamp", "precio_cierre"])
    df["fecha"] = pd.to_datetime(df["timestamp"], unit="ms").dt.strftime("%Y-%m-%d")
    df = df[["fecha", "precio_cierre"]].drop_duplicates("fecha")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Descarga el histórico de precio de BTC.")
    parser.add_argument("--fuente", choices=["coingecko", "yfinance"], default="coingecko")
    parser.add_argument("--dias", type=int, default=60, help="Solo para CoinGecko.")
    parser.add_argument("--inicio", default="2026-09-01", help="Solo para yfinance.")
    parser.add_argument("--fin", default="2026-10-20", help="Solo para yfinance.")
    parser.add_argument("--output", default="data/raw/precio_btc.csv")
    args = parser.parse_args()

    if args.fuente == "yfinance":
        df = descargar_yfinance(args.inicio, args.fin)
    else:
        df = descargar_coingecko(args.dias)

    salida = Path(args.output)
    salida.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(salida, index=False)
    print(f"Guardadas {len(df)} fechas de precio -> {salida}")
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
