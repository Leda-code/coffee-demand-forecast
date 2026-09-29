"""Distribuição do total previsto entre produtos (abordagem top-down) e alternativa bottom-up.

Observação importante: com médias móveis, o bottom-up de N semanas (cada produto com sua
própria média) é matematicamente igual a top-down com total = média de N semanas e mix
estimado na mesma janela de N semanas. Por isso a pergunta útil é a janela usada para o mix:
o TOTAL muda de nível rápido (janela curta), enquanto o MIX deveria ser mais estável
(janela longa).
"""
from __future__ import annotations

import pandas as pd


def _normalize(counts: pd.DataFrame) -> pd.DataFrame:
    return counts.div(counts.sum(axis=1), axis=0)


def trailing_shares(wp: pd.DataFrame, weeks: int) -> pd.DataFrame:
    """Participação de cada produto nas últimas `weeks` semanas (sem incluir a semana atual)."""
    return _normalize(wp.rolling(weeks).sum().shift(1))


def expanding_shares(wp: pd.DataFrame) -> pd.DataFrame:
    """Participação de cada produto em todo o histórico até a semana anterior."""
    return _normalize(wp.expanding().sum().shift(1))


def top_down(total_forecast: pd.Series, shares: pd.DataFrame) -> pd.DataFrame:
    """Previsão por produto = total previsto x participação estimada do produto."""
    return shares.mul(total_forecast, axis=0)


def bottom_up_ma(wp: pd.DataFrame, weeks: int = 4) -> pd.DataFrame:
    """Previsão por produto usando a média móvel de cada produto separadamente."""
    return wp.shift(1).rolling(weeks).mean()


def pooled_wape(actual: pd.DataFrame, forecast: pd.DataFrame) -> float:
    """WAPE agregado de todos os produtos e semanas."""
    return float((actual - forecast).abs().sum().sum() / actual.sum().sum())


def wape_by_product(actual: pd.DataFrame, forecast: pd.DataFrame) -> pd.Series:
    return (actual - forecast).abs().sum() / actual.sum()
