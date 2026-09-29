# Previsão de demanda para reposição de uma máquina de café

Case técnico de Ciência de Dados. Objetivo: prever a demanda de uma máquina de café para apoiar a decisão de reposição (evitar falta de produto e sobra de estoque).

## Dados

Kaggle, dataset [Coffee Sales](https://www.kaggle.com/datasets/ihelon/coffee-sales). Este projeto usa apenas `index_1.csv` (uma máquina, 01/03/2024 a 23/03/2025, 3.636 vendas, 8 produtos). O `index_2.csv` é de **outra máquina** (outro catálogo, 30 produtos) e não foi usado na modelagem; ele entra na discussão de arquitetura (um modelo por máquina).

Os dados não são versionados. Baixe o arquivo e salve em `data/raw/index_1.csv`.

## Como executar

```bash
python -m venv .venv
.venv\Scripts\activate          # Mac/Linux: source .venv/bin/activate
pip install -r requirements.txt
pytest -q                       # testes unitários

python scripts/01_explore.py data/raw/index_1.csv          # carga, limpeza, diagnóstico
python scripts/02_eda.py data/raw/index_1.csv              # gráficos exploratórios
python scripts/03_baselines.py data/raw/index_1.csv --test-start 2025-01-20
python scripts/04_rolling_eval.py data/raw/index_1.csv     # avaliação em blocos de 4 semanas
python scripts/05_weekly_forecast.py data/raw/index_1.csv  # previsão semanal e reposição simulada
python scripts/06_product_mix.py data/raw/index_1.csv      # divisão por produto
```

Gráficos e tabelas são gravados em `reports/`.

## Estrutura

- `src/coffee_forecast/` código reutilizável: `data.py` (carga, limpeza, agregação), `baselines.py` (baselines diários e métricas), `models.py` (nível × dia da semana), `weekly.py` (agregação semanal e simulação de reposição), `mix.py` (divisão por produto)
- `scripts/` etapas executáveis do pipeline, numeradas
- `tests/` testes unitários (inclui verificações de ausência de vazamento do futuro)
- `reports/` figuras e tabelas geradas

## Decisões e resultados

**Achados dos dados:** cerca de 9 vendas por dia; demanda por produto intermitente (Espresso sem venda em 73% dos dias); mudança de regime em fev/2025 (média semanal de ~37 para ~106 unidades); padrão por dia da semana só aparece com volume alto; participação dos produtos instável (Americano de 7% a 31% entre blocos de 4 semanas).

**Decisões:** prever o total por semana (alinhado à reposição); métrica WAPE no lugar de MAPE (dias com poucas ou zero vendas); validação temporal em blocos de 4 semanas; toda previsão usa apenas dados anteriores ao período previsto.

| Etapa | Melhor abordagem | Resultado |
|---|---|---|
| Total diário, 1 dia à frente | nível de 7 dias × índice do dia da semana | MAE 3,65; WAPE 37% (~3% melhor que média móvel de 7 dias; ~25% melhor que "ontem") |
| Total semanal, 1 semana à frente | média de 2 semanas | MAE 14,4; WAPE 21% (média de 4 semanas: 25%) |
| Reposição simulada (folga 0% / 10% / 20%) | média de 2 semanas | demanda atendida 88% / 92% / 94%; estoque não vendido 9% / 14% / 20% |
| Por produto, semanal | média de 4 semanas por produto | WAPE agregado 41% (mix de 8 semanas: 43%; mix histórico: 46%) |

## Limitações

- Um ano de dados e uma máquina; sem sazonalidade anual estimável.
- As vendas históricas já foram limitadas pelo estoque real, então a demanda verdadeira pode ser maior.
- A simulação de reposição assume um reabastecimento por semana e não considera o custo de falta contra o de sobra.
- Diferenças pequenas entre modelos (por exemplo, ~3% no diário) não foram testadas estatisticamente.

## Arquitetura proposta para produção

Vendas da máquina → ingestão diária (batch) → data lake (Databricks/Azure) → job semanal que agrega e prevê → tabela de previsão por máquina → dashboard e alerta de reposição. Monitoramento do WAPE semanal com alerta fora da faixa histórica, código versionado em Git e registro de modelo para rastrear versões. Um modelo por máquina.

## Próximos passos

Agrupar produtos em famílias, detectar mudança de regime automaticamente, testar modelos com lags (por exemplo LightGBM) e variáveis externas, incluir o custo de falta e de sobra para escolher a folga, e testar a significância dos ganhos.
