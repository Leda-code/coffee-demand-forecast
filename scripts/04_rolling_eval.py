"""Passo 4: compara baselines e o modelo nível x índice semanal em blocos de 4 semanas.

Em vez de um único corte de teste (frágil com mudanças de regime), avalia todo o período
após o aquecimento (primeiras 8 semanas de histórico) e mostra o WAPE por bloco de 28 dias.

Uso:
    python scripts/04_rolling_eval.py data/raw/index_1.csv
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coffee_forecast.baselines import bias, mae, one_step_forecasts, wape  # noqa: E402
from coffee_forecast.data import clean, load_raw, to_daily  # noqa: E402
from coffee_forecast.models import level_weekday_forecast  # noqa: E402

WARMUP_DAYS = 56
BLOCK_DAYS = 28


def main(path: str) -> None:
    df = clean(load_raw(path))
    s = to_daily(df, by_product=False).set_index("date")["units"]

    fc = one_step_forecasts(s)
    fc["nivel 7d x indice semanal (8 sem.)"] = level_weekday_forecast(s, shrink=1.0)
    fc["nivel 7d x indice semanal (8 sem., encolhido 50%)"] = level_weekday_forecast(s, shrink=0.5)

    start = s.index[WARMUP_DAYS]
    y = s.loc[start:]
    fc = fc.loc[start:]
    print(f"Avaliação de {start.date()} a {s.index[-1].date()} ({len(y)} dias)\n")

    rows = []
    for name in fc.columns:
        pair = pd.concat([y, fc[name]], axis=1).dropna()
        a, b = pair.iloc[:, 0], pair.iloc[:, 1]
        rows.append({"modelo": name, "MAE": mae(a, b), "WAPE": wape(a, b), "vies": bias(a, b)})
    overall = pd.DataFrame(rows).sort_values("MAE").reset_index(drop=True)
    print("=== PERÍODO COMPLETO ===")
    print(overall.round(3).to_string(index=False), "\n")

    block_id = ((y.index - start).days // BLOCK_DAYS)
    wape_blocks = {}
    for name in fc.columns:
        vals = []
        for b_id in sorted(set(block_id)):
            m = block_id == b_id
            pair = pd.concat([y[m], fc[name][m]], axis=1).dropna()
            vals.append(wape(pair.iloc[:, 0], pair.iloc[:, 1]))
        wape_blocks[name] = vals
    labels = [
        (start + pd.Timedelta(days=int(b) * BLOCK_DAYS)).strftime("%d/%m/%y")
        for b in sorted(set(block_id))
    ]
    blocks = pd.DataFrame(wape_blocks, index=labels).T
    print("=== WAPE POR BLOCO DE 28 DIAS (coluna = data de início do bloco) ===")
    print((blocks * 100).round(0).astype(int).to_string(), "\n")

    wins = blocks.idxmin().value_counts()
    print("Modelo com menor WAPE em cada bloco (contagem de blocos vencidos):")
    print(wins.to_string(), "\n")

    out = Path("reports")
    (out / "figures").mkdir(parents=True, exist_ok=True)
    overall.to_csv(out / "04_metricas_periodo_completo.csv", index=False)
    blocks.to_csv(out / "04_wape_por_bloco.csv")
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for name in blocks.index:
        ax.plot(labels, blocks.loc[name] * 100, marker="o", label=name)
    ax.set(title="WAPE (%) por bloco de 28 dias", ylabel="WAPE %", xlabel="início do bloco")
    ax.legend(fontsize=7)
    plt.setp(ax.get_xticklabels(), rotation=45)
    fig.tight_layout()
    fig.savefig(out / "figures" / "07_wape_por_bloco.png", dpi=130)
    print("Arquivos salvos em reports/ (CSVs e figures/07_wape_por_bloco.png)")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
