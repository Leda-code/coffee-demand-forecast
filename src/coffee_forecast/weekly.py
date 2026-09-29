"""Agregação semanal, baselines de previsão semanal e simulação simples de reposição.

Semana = segunda a domingo. Só semanas completas (7 dias) entram na análise.
Previsão de 1 semana à frente: para prever a semana t usa apenas semanas até t-1.
"""
from __future__ import annotations

import pandas as pd


def _week_start(dates: pd.DatetimeIndex | pd.Series):
    """Segunda-feira da semana de cada data."""
    if isinstance(dates, pd.Series):
        return dates - pd.to_timedelta(dates.dt.dayofweek, unit="D")
    return dates - pd.to_timedelta(dates.dayofweek, unit="D")


def complete_weeks(daily_total: pd.Series) -> pd.Series:
    """Soma semanal (seg-dom) da série diária, descartando semanas incompletas."""
    ws = pd.Series(_week_start(daily_total.index), index=daily_total.index, name="week")
    grouped = daily_total.groupby(ws.to_numpy())
    sums, sizes = grouped.sum(), grouped.size()
    out = sums[sizes == 7]
    out.index = pd.DatetimeIndex(out.index, name="week")
    out.name = "units"
    return out


def weekly_by_product(df: pd.DataFrame, weeks: pd.DatetimeIndex) -> pd.DataFrame:
    """Unidades por semana e produto (linhas = semanas, colunas = produtos)."""
    ws = _week_start(df["date"]).rename("week")
    wp = df.groupby([ws, df["coffee_name"]]).size().unstack(fill_value=0)
    return wp.reindex(weeks, fill_value=0).astype(float)


def weekly_baselines(w: pd.Series) -> pd.DataFrame:
    """Previsões de 1 semana à frente (usam só semanas anteriores)."""
    prev = w.shift(1)
    return pd.DataFrame(
        {
            "ultima semana": prev,
            "media 2 semanas": prev.rolling(2).mean(),
            "media 4 semanas": prev.rolling(4).mean(),
            "suavizacao exponencial (alfa=0.5)": prev.ewm(alpha=0.5, adjust=False).mean(),
        }
    )


def replenishment_sim(actual: pd.Series, forecast: pd.Series, buffer: float = 0.0) -> dict:
    """Simulação ilustrativa: estoque da semana = previsão x (1 + buffer).

    - atendimento: fração da demanda semanal coberta pelo estoque (1 - falta/demanda)
    - sobra: fração do estoque que não foi vendida (proxy de desperdício)

    Limitação: as vendas históricas já foram limitadas pelo estoque real da máquina,
    então a demanda verdadeira pode ser maior que a observada.
    """
    stock = forecast * (1.0 + buffer)
    shortfall = (actual - stock).clip(lower=0).sum()
    surplus = (stock - actual).clip(lower=0).sum()
    return {
        "atendimento": float(1.0 - shortfall / actual.sum()),
        "sobra": float(surplus / stock.sum()),
    }
