import numpy as np
import pandas as pd

from coffee_forecast.mix import bottom_up_ma, expanding_shares, top_down, trailing_shares
from coffee_forecast.weekly import (
    complete_weeks,
    replenishment_sim,
    weekly_baselines,
    weekly_by_product,
)


def _daily(start="2024-03-01", n=30):  # 01/03/2024 é sexta-feira
    idx = pd.date_range(start, periods=n, freq="D")
    return pd.Series(1.0, index=idx)


def test_complete_weeks_drops_partial_weeks():
    w = complete_weeks(_daily("2024-03-01", 30))  # sex 01/03 ... 30 dias
    assert (w == 7).all()
    assert w.index[0] == pd.Timestamp("2024-03-04")  # primeira segunda completa
    assert w.index[-1] == pd.Timestamp("2024-03-18")  # 25-30/03 tem só 6 dias: descartada


def test_weekly_by_product_counts():
    dates = pd.to_datetime(["2024-03-04", "2024-03-05", "2024-03-11"])
    df = pd.DataFrame({"date": dates, "coffee_name": ["A", "B", "A"]})
    weeks = pd.DatetimeIndex(["2024-03-04", "2024-03-11"])
    wp = weekly_by_product(df, weeks)
    assert wp.loc["2024-03-04", "A"] == 1 and wp.loc["2024-03-04", "B"] == 1
    assert wp.loc["2024-03-11", "A"] == 1 and wp.loc["2024-03-11", "B"] == 0


def test_weekly_baselines_no_leakage():
    w = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    fc = weekly_baselines(w)
    assert fc["ultima semana"].iloc[3] == 30.0
    assert fc["media 2 semanas"].iloc[3] == 25.0


def test_replenishment_sim_values():
    actual = pd.Series([10.0, 10.0])
    forecast = pd.Series([8.0, 14.0])
    r = replenishment_sim(actual, forecast, buffer=0.0)
    assert np.isclose(r["atendimento"], 1 - 2 / 20)  # faltaram 2 de 20
    assert np.isclose(r["sobra"], 4 / 22)  # sobraram 4 de 22 em estoque


def test_bottom_up_sum_equals_total_moving_average():
    rng = np.random.default_rng(1)
    idx = pd.date_range("2024-03-04", periods=20, freq="W-MON")
    wp = pd.DataFrame(rng.integers(0, 10, size=(20, 3)).astype(float), index=idx, columns=list("ABC"))
    bu = bottom_up_ma(wp, 4)
    total_ma = wp.sum(axis=1).shift(1).rolling(4).mean()
    ok = bu.dropna().index
    assert np.allclose(bu.loc[ok].sum(axis=1), total_ma.loc[ok])


def test_shares_sum_to_one_and_top_down_preserves_total():
    idx = pd.date_range("2024-03-04", periods=12, freq="W-MON")
    wp = pd.DataFrame(np.arange(1, 37, dtype=float).reshape(12, 3), index=idx, columns=list("ABC"))
    for shares in (trailing_shares(wp, 4), expanding_shares(wp)):
        valid = shares.dropna()
        assert np.allclose(valid.sum(axis=1), 1.0)
    total_fc = pd.Series(100.0, index=idx)
    td = top_down(total_fc, trailing_shares(wp, 4)).dropna()
    assert np.allclose(td.sum(axis=1), 100.0)


def test_trailing_shares_do_not_use_current_week():
    idx = pd.date_range("2024-03-04", periods=8, freq="W-MON")
    wp = pd.DataFrame(1.0, index=idx, columns=list("AB"))
    base = trailing_shares(wp, 4).iloc[-1]
    wp2 = wp.copy()
    wp2.iloc[-1] = [1000.0, 0.0]
    assert np.allclose(base, trailing_shares(wp2, 4).iloc[-1])
