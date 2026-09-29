import pandas as pd

from coffee_forecast.baselines import bias, evaluate, mae, one_step_forecasts, wape


def _series(n=60):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.Series([10 + (i % 7) for i in range(n)], index=idx, dtype=float)


def test_metrics():
    y = pd.Series([10.0, 10.0])
    yhat = pd.Series([12.0, 8.0])
    assert mae(y, yhat) == 2.0
    assert wape(y, yhat) == 0.2
    assert bias(y, yhat) == 0.0


def test_no_leakage_naive_uses_previous_day():
    s = _series()
    fc = one_step_forecasts(s)
    assert fc["ingenuo (ontem)"].iloc[10] == s.iloc[9]


def test_seasonal_naive_is_perfect_on_weekly_pattern():
    s = _series()
    res = evaluate(s, s.index[35])
    row = res[res["modelo"].str.startswith("sazonal")].iloc[0]
    assert row["MAE"] == 0.0
