"""Passo 6: previsão por produto — total previsto x mix (top-down) vs média por produto (bottom-up).

Uso:
    python scripts/06_product_mix.py data/raw/index_1.csv

Todas as abordagens usam a mesma previsão de total (média de 4 semanas). O que muda é como
o total é dividido entre produtos. O bottom-up de 4 semanas equivale a um top-down com mix
estimado nas últimas 4 semanas; os demais testam mix mais estável (8 semanas e histórico todo).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coffee_forecast.data import clean, load_raw, to_daily  # noqa: E402
from coffee_forecast.mix import (  # noqa: E402
    bottom_up_ma,
    expanding_shares,
    pooled_wape,
    top_down,
    trailing_shares,
    wape_by_product,
)
from coffee_forecast.weekly import complete_weeks, weekly_baselines, weekly_by_product  # noqa: E402

WARMUP_WEEKS = 8
BLOCK_WEEKS = 4


def main(path: str) -> None:
    df = clean(load_raw(path))
    daily = to_daily(df, by_product=False).set_index("date")["units"]
    wt = complete_weeks(daily)
    wp = weekly_by_product(df, wt.index)

    total_fc = weekly_baselines(wt)["media 4 semanas"]
    methods = {
        "bottom-up (media 4 sem. por produto)": bottom_up_ma(wp, 4),
        "top-down (mix 8 semanas)": top_down(total_fc, trailing_shares(wp, 8)),
        "top-down (mix historico todo)": top_down(total_fc, expanding_shares(wp)),
    }

    start = wp.index[WARMUP_WEEKS]
    actual = wp.loc[start:]
    print(f"Avaliação de {start.date()} a {actual.index[-1].date()} ({len(actual)} semanas)\n")

    pooled = pd.Series({k: pooled_wape(actual, v.loc[start:]) for k, v in methods.items()})
    print("=== WAPE AGREGADO (todos os produtos) ===")
    print((pooled * 100).round(1).to_string(), "\n")

    by_prod = pd.DataFrame({k: wape_by_product(actual, v.loc[start:]) for k, v in methods.items()})
    by_prod.insert(0, "media_sem", actual.mean().round(1))
    by_prod = by_prod.sort_values("media_sem", ascending=False)
    show = by_prod.copy()
    show.iloc[:, 1:] = (show.iloc[:, 1:] * 100).round(0)
    print("=== WAPE (%) POR PRODUTO (media_sem = unidades médias por semana) ===")
    print(show.to_string(), "\n")

    block = (pd.Series(range(len(wp)), index=wp.index) // BLOCK_WEEKS)
    shares = wp.groupby(block.to_numpy()).sum()
    shares = shares.div(shares.sum(axis=1), axis=0) * 100
    shares.index = [wp.index[int(i) * BLOCK_WEEKS].strftime("%d/%m/%y") for i in shares.index]
    order = wp.sum().sort_values(ascending=False).index
    print("=== MIX (% das unidades) POR BLOCO DE 4 SEMANAS (início do bloco) ===")
    print(shares[order].round(0).astype(int).to_string(), "\n")

    out = Path("reports")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for col in order:
        ax.plot(shares.index, shares[col], marker="o", label=col)
    ax.set(title="Participação de cada produto por bloco de 4 semanas (%)", ylabel="% das unidades")
    ax.legend(fontsize=7, ncol=2)
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    fig.savefig(out / "figures" / "09_mix_por_produto.png", dpi=130)
    by_prod.to_csv(out / "06_wape_por_produto.csv")
    print("Salvo: reports/figures/09_mix_por_produto.png e reports/06_wape_por_produto.csv")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
