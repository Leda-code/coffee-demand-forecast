"""Baselines simples de previsão com horizonte de 1 dia e métricas de erro.

Cada previsão para o dia t usa apenas informação até t-1 (sem vazamento do futuro).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y: pd.Series, yhat: pd.Series) -> float:
    return float(np.abs(y - yhat).mean())


def wape(y: pd.Series, yhat: pd.Series) -> float:
    """Erro absoluto total / vendas totais. Robusto a dias com poucas ou zero vendas
    (ao contrário do MAPE, que explode quando o real é zero ou muito pequeno)."""
    return float(np.abs(y - yhat).sum() / np.abs(y).sum())


def bias(y: pd.Series, yhat: pd.Series) -> float:
    """Erro médio com sinal: positivo = o modelo superestima a demanda."""
    return float((yhat - y).mean())


def one_step_forecasts(s: pd.Series) -> pd.DataFrame:
    """Previsões de baselines para cada dia da série diária contínua `s`."""
    prev = s.shift(1)
    same_weekday = pd.concat([s.shift(7 * k) for k in (1, 2, 3, 4)], axis=1)
    return pd.DataFrame(
        {
            "ingenuo (ontem)": prev,
            "sazonal ingenuo (mesmo dia sem. passada)": s.shift(7),
            "media movel 7d": prev.rolling(7).mean(),
            "media movel 28d": prev.rolling(28).mean(),
            "media mesmo dia da semana (4 sem.)": same_weekday.mean(axis=1),
        }
    )


def evaluate(s: pd.Series, test_start: str | pd.Timestamp) -> pd.DataFrame:
    """Métricas de cada baseline no período de teste (a partir de `test_start`)."""
    fc = one_step_forecasts(s).loc[test_start:]
    y = s.loc[test_start:]
    rows = []
    for name in fc.columns:
        pair = pd.concat([y, fc[name]], axis=1).dropna()
        rows.append(
            {
                "modelo": name,
                "MAE": mae(pair.iloc[:, 0], pair.iloc[:, 1]),
                "WAPE": wape(pair.iloc[:, 0], pair.iloc[:, 1]),
                "vies": bias(pair.iloc[:, 0], pair.iloc[:, 1]),
            }
        )
    return pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
