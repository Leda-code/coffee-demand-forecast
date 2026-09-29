"""Passo 5: previsão do total SEMANAL (horizonte de 1 semana) e simulação de reposição.

Uso:
    python scripts/05_weekly_forecast.py data/raw/index_1.csv

Por que semanal: a reposição de uma máquina não decide dia a dia. Prever a semana reduz o
peso do ruído diário e aproxima a previsão da decisão real.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coffee_forecast.baselines import bias, mae, wape  # noqa: E402
from coffee_forecast.data import clean, load_raw, to_daily  # noqa: E402
from coffee_forecast.weekly import (  # noqa: E402
    complete_weeks,
    replenishment_sim,
    weekly_baselines,
)

WARMUP_WEEKS = 8
BUFFERS = (0.0, 0.10, 0.20)


def main(path: str) -> None:
    df = clean(load_raw(path))
    daily = to_daily(df, by_product=False).set_index("date")["units"]
    wt = complete_weeks(daily)

    fc = weekly_baselines(wt)
    start = wt.index[WARMUP_WEEKS]
    y, fc = wt.loc[start:], fc.loc[start:]
    print(f"Semanas completas: {len(wt)} ({wt.index[0].date()} a {wt.index[-1].date()})")
    print(f"Avaliadas: {len(y)} semanas, de {start.date()} a {y.index[-1].date()}")
    print(f"Média semanal: {y.mean():.1f} unidades\n")

    rows = []
    for name in fc.columns:
        pair = pd.concat([y, fc[name]], axis=1).dropna()
        a, b = pair.iloc[:, 0], pair.iloc[:, 1]
        rows.append({"modelo": name, "MAE": mae(a, b), "WAPE": wape(a, b), "vies": bias(a, b)})
    res = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    print("=== PREVISÃO SEMANAL (1 semana à frente) ===")
    print(res.round(3).to_string(index=False), "\n")

    sim_rows = []
    for name in fc.columns:
        pair = pd.concat([y, fc[name]], axis=1).dropna()
        row = {"modelo": name}
        for buf in BUFFERS:
            r = replenishment_sim(pair.iloc[:, 0], pair.iloc[:, 1], buf)
            row[f"atend_{int(buf * 100)}%"] = round(r["atendimento"] * 100, 1)
            row[f"sobra_{int(buf * 100)}%"] = round(r["sobra"] * 100, 1)
        sim_rows.append(row)
    print("=== SIMULAÇÃO DE REPOSIÇÃO (estoque = previsão x (1 + folga)) ===")
    print("atend_X% = % da demanda atendida; sobra_X% = % do estoque não vendido; X = folga")
    print(pd.DataFrame(sim_rows).to_string(index=False), "\n")

    best = res.loc[0, "modelo"]
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(y, marker="o", label="real")
    ax.plot(fc[best], marker="o", label=f"previsto: {best}")
    ax.set(title="Total semanal: real vs previsto (melhor modelo)", ylabel="unidades/semana")
    ax.legend()
    out = Path("reports")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / "figures" / "08_previsao_semanal.png", dpi=130)
    res.to_csv(out / "05_metricas_semanais.csv", index=False)
    print("Salvo: reports/figures/08_previsao_semanal.png e reports/05_metricas_semanais.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
