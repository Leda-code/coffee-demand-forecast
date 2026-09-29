"""Passo 1: carrega os dados, limpa, agrega e imprime um diagnóstico.

Uso:
    python scripts/01_explore.py data/raw/index.csv
"""
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coffee_forecast.data import clean, load_raw, to_daily  # noqa: E402


def main(path: str) -> None:
    raw = load_raw(path)
    print("=== BRUTO ===")
    print(raw.head())
    print(raw.dtypes, "\n")
    print("Nulos por coluna:\n", raw.isna().sum(), "\n")

    df = clean(raw)
    print(f"=== LIMPO === linhas: {len(df)} (descartadas: {df.attrs['rows_dropped']})")
    print(f"Período: {df['date'].min().date()} -> {df['date'].max().date()}")
    print(f"Produtos: {df['coffee_name'].nunique()}")
    print(df["coffee_name"].value_counts(), "\n")

    total = to_daily(df, by_product=False)
    print("=== DIÁRIO (total) ===")
    print(total["units"].describe(), "\n")
    zero_days = (total["units"] == 0).sum()
    print(f"Dias sem nenhuma venda: {zero_days} de {len(total)}")

    print("\nMédia de unidades por dia da semana:")
    total["weekday"] = total["date"].dt.day_name()
    print(total.groupby("weekday")["units"].mean().round(1))

    print("\nUnidades por mês:")
    print(total.set_index("date")["units"].resample("MS").sum())

    out = Path("data/processed")
    out.mkdir(parents=True, exist_ok=True)
    to_daily(df).to_csv(out / "daily_by_product.csv", index=False)
    print(f"\nSalvo: {out / 'daily_by_product.csv'}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
