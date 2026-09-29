"""Passo 2: análise exploratória com gráficos e diagnósticos para decidir a modelagem.

Uso:
    python scripts/02_eda.py data/raw/index_1.csv

Gera figuras em reports/figures/ e imprime diagnósticos no terminal.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # salva em arquivo, sem abrir janela
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from coffee_forecast.data import clean, load_raw, to_daily  # noqa: E402

FIG_DIR = Path("reports/figures")


def zero_runs(daily: pd.DataFrame) -> pd.DataFrame:
    """Sequências consecutivas de dias sem venda (possível máquina fora do ar)."""
    is_zero = daily["units"].eq(0)
    group = (is_zero != is_zero.shift()).cumsum()
    runs = (
        daily[is_zero]
        .groupby(group[is_zero])["date"]
        .agg(inicio="min", fim="max", dias="count")
        .reset_index(drop=True)
    )
    return runs


def save(fig, name: str) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(FIG_DIR / name, dpi=130)
    plt.close(fig)
    print(f"  figura salva: {FIG_DIR / name}")


def main(path: str) -> None:
    df = clean(load_raw(path))
    total = to_daily(df, by_product=False)
    by_prod = to_daily(df, by_product=True)

    # 1) Dias sem venda: zero real ou máquina parada?
    print("=== SEQUÊNCIAS DE DIAS SEM VENDA ===")
    runs = zero_runs(total)
    print(runs.to_string(index=False) if len(runs) else "Nenhuma", "\n")

    # 2) Intermitência por produto: % de dias com zero venda
    print("=== INTERMITÊNCIA POR PRODUTO ===")
    summary = (
        by_prod.groupby("coffee_name")["units"]
        .agg(media_dia="mean", pct_dias_zero=lambda s: (s == 0).mean() * 100)
        .round(2)
        .sort_values("media_dia", ascending=False)
    )
    print(summary, "\n")

    print("=== FIGURAS ===")
    # 3) Série diária + média móvel de 7 dias
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(total["date"], total["units"], alpha=0.35, label="diário")
    ax.plot(total["date"], total["units"].rolling(7).mean(), label="média móvel 7d")
    ax.set(title="Unidades vendidas por dia", ylabel="unidades")
    ax.legend()
    save(fig, "01_serie_diaria.png")

    # 4) Série semanal
    weekly = total.set_index("date")["units"].resample("W").sum()
    fig, ax = plt.subplots(figsize=(11, 4))
    weekly.plot(ax=ax)
    ax.set(title="Unidades vendidas por semana", ylabel="unidades", xlabel="")
    save(fig, "02_serie_semanal.png")

    # 5) Vendas por hora do dia
    by_hour = df.groupby(df["datetime"].dt.hour).size()
    fig, ax = plt.subplots(figsize=(8, 4))
    by_hour.plot.bar(ax=ax)
    ax.set(title="Transações por hora do dia", xlabel="hora", ylabel="transações")
    save(fig, "03_por_hora.png")

    # 6) Média por dia da semana
    order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    wd = total.assign(wd=total["date"].dt.day_name()).groupby("wd")["units"].agg(["mean", "std"])
    wd = wd.reindex(order)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(wd.index, wd["mean"], yerr=wd["std"], capsize=3)
    ax.set(title="Média de unidades por dia da semana (± desvio)", ylabel="unidades")
    plt.setp(ax.get_xticklabels(), rotation=30)
    save(fig, "04_dia_da_semana.png")

    # 7) Participação de cada produto
    share = df["coffee_name"].value_counts()
    fig, ax = plt.subplots(figsize=(8, 4))
    share.sort_values().plot.barh(ax=ax)
    ax.set(title="Unidades vendidas por produto", xlabel="unidades")
    save(fig, "05_produtos.png")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    main(sys.argv[1])
