"""Passo 3: baselines de previsão do total diário de unidades.

Uso:
    python scripts/03_baselines.py data/raw/index_1.csv
    python scripts/03_baselines.py data/raw/index_1.csv data/raw/index_2.csv --test-start 2025-04-01

Sem --test-start, os últimos 60 dias da série são usados como teste.
Previsões com horizonte de 1 dia (prever amanhã com dados até hoje).
"""
import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coffee_forecast.baselines import evaluate, one_step_forecasts  # noqa: E402
from coffee_forecast.data import clean, load_raw, to_daily  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="um ou mais CSVs (ex.: index_1.csv index_2.csv)")
    ap.add_argument("--test-start", default=None, help="data inicial do teste (AAAA-MM-DD)")
    args = ap.parse_args()

    raw = pd.concat([load_raw(p) for p in args.paths], ignore_index=True)
    df = clean(raw)  # clean() já remove duplicatas exatas entre arquivos
    total = to_daily(df, by_product=False).set_index("date")["units"]

    test_start = pd.Timestamp(args.test_start) if args.test_start else total.index[-60]
    print(f"Série: {total.index.min().date()} -> {total.index.max().date()} ({len(total)} dias)")
    print(f"Treino/histórico: até {(test_start - pd.Timedelta(days=1)).date()}")
    print(f"Teste: {test_start.date()} -> {total.index.max().date()} "
          f"({(total.index >= test_start).sum()} dias)\n")

    res = evaluate(total, test_start)
    print(res.round(3).to_string(index=False))

    best = res.loc[0, "modelo"]
    fc = one_step_forecasts(total)[best].loc[test_start:]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(total.loc[test_start:], label="real")
    ax.plot(fc, label=f"previsto: {best}")
    ax.set(title="Período de teste: real vs melhor baseline", ylabel="unidades/dia")
    ax.legend()
    out = Path("reports/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "06_baseline_teste.png", dpi=130)
    print(f"\nFigura salva: {out / '06_baseline_teste.png'}")


if __name__ == "__main__":
    main()
