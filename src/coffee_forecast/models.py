"""Modelo de previsão de 1 dia: nível recente x índice do dia da semana.

Ideia: as vendas de amanhã = nível recente de vendas (média dos últimos 7 dias)
ajustado pelo padrão do dia da semana (razão entre a média daquele dia e a média geral,
estimada nas últimas `index_weeks` semanas). O nível se adapta rápido a mudanças de
regime; o índice captura o padrão semanal sem ficar preso a uma única observação.

`shrink` encolhe o índice em direção a 1 (0 = ignora dia da semana, 1 = usa o índice cheio),
para reduzir ruído quando há poucas semanas de histórico.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def level_weekday_forecast(
    s: pd.Series,
    level_window: int = 7,
    index_weeks: int = 8,
    shrink: float = 1.0,
) -> pd.Series:
    """Previsão para cada dia t usando apenas dados até t-1 (sem vazamento)."""
    if not isinstance(s.index, pd.DatetimeIndex):
        raise TypeError("A série precisa ter DatetimeIndex diário e contínuo.")
    hist_len = 7 * index_weeks
    out = pd.Series(np.nan, index=s.index)
    values = s.to_numpy(dtype=float)
    weekdays = s.index.dayofweek.to_numpy()

    for i in range(max(hist_len, level_window), len(s)):
        hist = values[i - hist_len : i]
        hist_wd = weekdays[i - hist_len : i]
        overall = hist.mean()
        level = values[i - level_window : i].mean()
        if overall == 0:
            out.iloc[i] = level
            continue
        day_mean = hist[hist_wd == weekdays[i]].mean()
        raw_idx = day_mean / overall
        idx = 1.0 + shrink * (raw_idx - 1.0)
        out.iloc[i] = level * idx
    return out
