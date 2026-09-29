import numpy as np
import pandas as pd

from coffee_forecast.models import level_weekday_forecast


def _weekly_pattern(n=100):
    idx = pd.date_range("2024-01-01", periods=n, freq="D")
    pattern = np.array([10, 12, 14, 16, 18, 8, 6], dtype=float)  # seg..dom
    return pd.Series([pattern[d.dayofweek] for d in idx], index=idx)


def test_exact_on_stable_weekly_pattern():
    s = _weekly_pattern()
    fc = level_weekday_forecast(s)
    valid = fc.dropna()
    assert len(valid) > 0
    assert np.allclose(valid, s.loc[valid.index])


def test_no_forecast_during_warmup():
    s = _weekly_pattern()
    fc = level_weekday_forecast(s, index_weeks=8)
    assert fc.iloc[:56].isna().all()


def test_shrink_zero_equals_level_only():
    s = _weekly_pattern()
    fc = level_weekday_forecast(s, shrink=0.0)
    expected = s.shift(1).rolling(7).mean()
    valid = fc.dropna().index
    assert np.allclose(fc.loc[valid], expected.loc[valid])


def test_no_leakage_future_values_do_not_change_past_forecast():
    s = _weekly_pattern()
    fc_full = level_weekday_forecast(s)
    s2 = s.copy()
    s2.iloc[-1] = 999.0
    fc_changed = level_weekday_forecast(s2)
    assert fc_full.iloc[-1] == fc_changed.iloc[-1]
