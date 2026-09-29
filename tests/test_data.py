import pandas as pd
import pytest

from coffee_forecast.data import clean, load_raw, to_daily


def _sample() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2024-03-01"] * 4 + ["2024-03-03"],
            "datetime": [
                "2024-03-01 10:00:00",
                "2024-03-01 10:00:00",  # duplicata exata
                "2024-03-01 11:00:00",
                "2024-03-01 12:00:00",  # valor inválido
                "2024-03-03 09:00:00",
            ],
            "cash_type": ["card"] * 5,
            "card": ["A", "A", "B", "C", "D"],
            "money": [30.0, 30.0, 25.0, -1.0, 28.0],
            "coffee_name": ["Latte", "Latte", "Espresso", "Latte", "Latte"],
        }
    )


def test_load_raw_missing_columns(tmp_path):
    p = tmp_path / "x.csv"
    pd.DataFrame({"a": [1]}).to_csv(p, index=False)
    with pytest.raises(ValueError):
        load_raw(p)


def test_clean_removes_duplicates_and_invalid():
    out = clean(_sample())
    assert len(out) == 3
    assert (out["money"] > 0).all()


def test_to_daily_fills_missing_days_with_zero():
    daily = to_daily(clean(_sample()), by_product=False)
    assert len(daily) == 3  # 01, 02, 03
    assert daily.loc[daily["date"] == "2024-03-02", "units"].iloc[0] == 0


def test_to_daily_by_product_has_full_grid():
    daily = to_daily(clean(_sample()), by_product=True)
    assert len(daily) == 3 * 2  # 3 dias x 2 produtos
