# Registro de decisões

Uma entrada por decisão. Formato: contexto → decisão → consequência. Mais recente no topo.

---

## 2026-10-27: Toolchain e definição da população

**Contexto.** O plano da semana 1 sugeria `requirements.txt` + venv. O arquivo de dados tem
~1,6 GB, grande demais para carregar inteiro no pandas com conforto.

**Decisão.**
- `uv` + `pyproject.toml` no lugar de `requirements.txt` (lockfile reproduzível, um só comando).
- Python 3.14 (o do sistema; o uv aqui não baixa interpretadores automaticamente).
- DuckDB para ler o parquet com filtro e projeção; só agregados vão para o pandas.
- População fixada antes de olhar qualquer taxa: `game_mode = 'Normal'`,
  `match_mode IN ('Unranked', 'Ranked')`, `match_outcome = 'TeamWin'` (em `config/analysis.toml`).
  Em 15/10/2026 isso manteve 87,4% das linhas.

**Consequência.** Mudanças na população exigem nova entrada aqui, com o motivo.
