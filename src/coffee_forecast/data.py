"""Carga, validação e agregação dos dados de vendas da máquina de café.

Dataset: Kaggle "Coffee Sales" (ihelon). Cada linha é uma transação (1 café vendido).
Colunas esperadas: date, datetime, cash_type, card, money, coffee_name.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = {"date", "datetime", "money", "coffee_name"}


def load_raw(path: str | Path) -> pd.DataFrame:
    """Lê o CSV bruto e valida o esquema mínimo."""
    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Colunas ausentes no arquivo: {sorted(missing)}")
    return df


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza tipos e remove registros inválidos.

    Decisões (documentar na apresentação):
    - Duplicatas exatas são removidas (mesmo timestamp, produto, valor e cartão).
    - Linhas sem produto ou com valor não positivo são descartadas.
    """
    out = df.copy()
    out["datetime"] = pd.to_datetime(out["datetime"], errors="coerce")
    out["money"] = pd.to_numeric(out["money"], errors="coerce")
    out["coffee_name"] = out["coffee_name"].astype("string").str.strip()

    before = len(out)
    out = out.dropna(subset=["datetime", "money", "coffee_name"])
    out = out[out["money"] > 0]
    out = out.drop_duplicates()
    out["date"] = out["datetime"].dt.normalize()

    out.attrs["rows_dropped"] = before - len(out)
    return out.sort_values("datetime").reset_index(drop=True)


def to_daily(df: pd.DataFrame, by_product: bool = True) -> pd.DataFrame:
    """Agrega para granularidade diária (unidades vendidas e receita).

    Reindexa para o calendário completo, preenchendo dias sem venda com 0.
    Atenção: assume que dia sem registro = zero vendas (e não máquina fora do ar).
    """
    full_range = pd.date_range(df["date"].min(), df["date"].max(), freq="D")

    if by_product:
        daily = (
            df.groupby(["date", "coffee_name"])
            .agg(units=("money", "size"), revenue=("money", "sum"))
            .reset_index()
        )
        products = daily["coffee_name"].unique()
        idx = pd.MultiIndex.from_product([full_range, products], names=["date", "coffee_name"])
        daily = daily.set_index(["date", "coffee_name"]).reindex(idx, fill_value=0).reset_index()
    else:
        daily = df.groupby("date").agg(units=("money", "size"), revenue=("money", "sum"))
        daily = daily.reindex(full_range, fill_value=0).rename_axis("date").reset_index()

    return daily
