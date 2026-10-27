# winlock

**A taxa de vitória de um herói de Deadlock é confiável, ou é ruído estatístico?**

Análise da win rate por herói com intervalos de confiança de Wilson, a partir de dados públicos da
[Deadlock API](https://api.deadlock-api.com/docs).

## Dados

- Fonte: snapshot residual da tabela `match_player` (uma linha por jogador por partida).
- Arquivo: `data/raw/20261020T020300Z-20261020T221300Z-50a132fe33c6.parquet` (~1,6 GB, fora do git).
- Período: janela de `created_at` de 20/10/2026, 02:03–22:13 UTC. *Período por `start_time`: a confirmar.*
- Patch / versão do jogo: *a preencher.*

## Como rodar

Requer [uv](https://docs.astral.sh/uv/).

```sh
make setup   # instala dependências e o filtro nbstripout
make test    # testes
make lint    # ruff
make nb      # abre o Jupyter Lab
```

## Método

*A preencher.*

## O que foi descartado e por quê

*A preencher (filtros de população, duplicatas, partidas incompletas, outliers).*

## Resultado

*A preencher.*

## Limitações

*A preencher.*
