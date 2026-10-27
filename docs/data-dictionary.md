# Dicionário de dados: `match_player`

Uma linha por jogador por partida (157 colunas). Marque **verificado** só depois de conferir nos
dados, com a consulta usada em "Como verifiquei".

| Coluna | Tipo | Significado | Verificado? | Como verifiquei / observação |
|---|---|---|---|---|
| `match_id` | | ID da partida; particionamento `intDiv(match_id, 1000000)` | a confirmar | 12 linhas por partida? |
| `account_id` | | Identifica o jogador. **Dado pessoal: nunca publicar** | a confirmar | |
| `hero_id` | | ID do herói (nomes em `data/reference/heroes.csv`) | a confirmar | |
| `team` | | Time do jogador | a confirmar | |
| `won` | bool | Jogador venceu a partida | a confirmar | |
| `start_time` | | Início da partida; usar para definir dias | a confirmar | Há cauda antiga? |
| `created_at` | | Quando a linha entrou no banco (marca d'água) | a confirmar | |
| `duration_s` | | Duração da partida em segundos | a confirmar | Outliers? |
| `game_mode` | | Filtro: `'Normal'` | a confirmar | |
| `match_mode` | | Filtro: `'Unranked'`, `'Ranked'` | a confirmar | |
| `match_outcome` | | Filtro: `'TeamWin'` | a confirmar | |
| `party` | | Grupo; 0 = solo (manifest), anulável | a confirmar | Preenchida nas partidas recentes? |
| `average_badge` | | Faixa de habilidade; só em ranked (~14%) | a confirmar | |
| `game_mode_version` | | Possível proxy de patch: **não documentado, não assumir** | a confirmar | |
