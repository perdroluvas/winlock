 Win rate entre dias: sinal ou ruído?
 Documento de contexto do projeto Deadlock · versão de 22/10/2026 · extensão da Semana 1
 (não substitui o gráfico principal)

 1. A proposta e como afiná-la
 A ideia original: usar o residual de 20/10 para calcular a win rate por herói e comparar com o dia
 seguinte. A comparação é boa e responde à pergunta que guia o projeto (“a taxa de vitória é
 confiável ou é ruído?”), mas comparar win rates entre dias, sozinho, é descritivo: mostra
 heróis subindo e descendo sem dizer se é sinal ou ruído.

  Pergunta mensurável: quanto da diferença de win rate entre dois dias é só ruído binomial, e
  quanto é mudança real (patch, meta, composição de jogadores)?


 2. Método principal: z-score por herói com grupo de controle
 Para cada herói, calcule a diferença entre o dia 1 e o dia 2, padronizada pelo erro esperado se nada
 tivesse mudado (p = taxa de vitória combinada dos dois dias):

  z = (p1 - p2) / sqrt( p * (1 - p) * (1/n1 + 1/n2) )
  p = (vitorias1 + vitorias2) / (n1 + n2)

  • Se só houvesse ruído binomial, os z de todos os heróis seguiriam uma normal padrão (desvio
    padrão ≈ 1).
  • Desvio padrão dos z bem acima de 1: há mudança real entre os dias. Perto de 1: a “variação” é
    quase toda ruído.
  • Grupo de controle sem dado novo: divida o dia 1 em duas metades por match_id % 2
    (partidas inteiras, nunca linhas soltas) e calcule os mesmos z entre as metades. Isso mostra como
    é o ruído puro nos seus dados. Compare o desvio padrão dos z entre dias com o das metades.
  • O controle importa porque jogadores repetidos e a estrutura de times deixam o ruído real um
    pouco maior que o binomial teórico. A referência empírica corrige isso.

 Dois cuidados obrigatórios
  • Várias comparações: são dezenas de heróis testados de uma vez. Aplique correção (Holm ou
    Benjamini-Hochberg) antes de dizer que um herói “mudou”.
  • Significativo não é relevante: com ~1M de linhas por dia, o erro padrão por herói fica em
    torno de 0,3 ponto percentual (estimativa: depende do número de heróis). Diferenças minúsculas
    aparecerão como significativas. Reporte sempre o tamanho da diferença, não só o p-valor.


 3. Riscos do desenho (o que checar antes de comparar)
 Risco                      Por que importa                                O que fazer

 Janela por                 O residual é recortado pelo momento em         Definir os dois dias por start_time. O
 created_at                 que a linha entrou no banco, não pelo da       script de inspeção mostra se há cauda
                            partida. Pode trazer partidas atrasadas de     antiga.
                            datas variadas.

 Mesma janela de            O perfil de jogadores muda ao longo do dia e   Comparar o mesmo intervalo de horas
 horas                      pode mexer na win rate. O arquivo cobre        nos dois dias.
                            02:03 a 22:13 UTC.

 Patch entre os dias        Se houve patch, a pergunta vira “o patch       Conferir “Patch Changes” no site e
                            mudou os heróis?”, que mistura efeito do       declarar no README qual pergunta
                            patch com adaptação dos jogadores.             está sendo respondida.




Projeto Deadlock · Contexto: estabilidade da win rate entre dias                                          Página 1
 Risco                      Por que importa                                     O que fazer

 Disponibilidade do         O residual foi construído cerca de um dia           Se precisar, usar o Playground
 dia seguinte               após a janela. O arquivo do dia seguinte            filtrando por faixa de match_id.
                            pode não existir ainda (inferência a
                            confirmar no manifest).

 Erro padrão otimista       A fórmula assume observações                        Registrar como limitação; o controle
                            independentes, mas o mesmo jogador                  por metades ajuda a calibrar.
                            aparece várias vezes e cada partida tem 6
                            vitórias e 6 derrotas.


 4. Análises mais incomuns com os mesmos dados
 #    Análise                                              Colunas                 Observação

 1    Win rate por duração da partida: heróis              duration_s, won         A duração é resultado do jogo,
      que “escalam” (ganham mais em jogos                                          então é descritivo, não causal.
      longos) versus os que ganham cedo.

 2    Win rate por hora do dia: efeito da                  start_time, won         Risco alto de falso positivo (herói
      composição de jogadores ao longo do dia.                                     × hora): exige correção de
                                                                                   múltiplas comparações.

 3    Solo versus grupo.                                   party                   Coluna anulável, 0 = solo segundo
                                                                                   o manifest. Conferir se vem
                                                                                   preenchida nas partidas recentes.

 4    Dia 1 prevê dia 2? Usar a win rate do dia            hero_id, won, team      Já é o embrião da Fase 2 (modelo).
      1 para prever vencedores do dia 2.                                           Split por tempo evita vazamento.


 5. Recomendação de escopo
  Prioridade: o plano da Semana 1 é um dia de dados, o gráfico com IC de Wilson e o post. A
  comparação entre dias é uma boa extensão, mas só depois que o gráfico base existir. Pode virar a
  seção “estabilidade” do mesmo post ou um segundo post. Não troque o gráfico principal por ela.

  Semana 2: duração, hora do dia, party e faixa de habilidade.

  Próximo passo: rodar o script de inspeção no residual correto (~1,5 GB) e conferir se as datas de
  start_time formam uma janela limpa.
 Aviso sobre o gancho do post: com ~1M de linhas, os intervalos de confiança ficam muito estreitos para heróis
 populares, então “55% pode ser ruído” aparece com menos clareza. Ângulos honestos: contrastar uma amostra
 pequena com a completa, ou mostrar que algumas diferenças são significativas mas pequenas demais para
 importar na prática.


 Anexo: fatos sobre os dados (verificados e a confirmar)
 Verificados em arquivos e no manifest durante a conversa:
  • Tabela match_player: uma linha por jogador por partida, 157 colunas. won já vem pronta
     (booleano).
  • Filtros definidos para a população do post: game_mode = 'Normal', match_mode IN
     ('Unranked','Ranked'), match_outcome = 'TeamWin'. Em 15/10/2026 isso manteve 87,4% das
     linhas (1.384.253 de 1.583.544); StreetBrawl era 12,1%. Defina a população antes de ver as
     taxas.
  • average_badge só existe em partidas ranqueadas (~14% do dia). Análise por habilidade vale só
     para esse subconjunto.
  • Particionamento intDiv(match_id, 1000000); marca d’água created_at. Arquivos delta e
     residual são recortados por created_at.



Projeto Deadlock · Contexto: estabilidade da win rate entre dias                                                   Página 2
  • O arquivo 20261019T135700Z-ebb3ba206178 enviado antes tinha match_id 6669 (abril de 2024):
    não é o residual de 20/10.

 Inferências ainda não confirmadas:
  • O residual pode conter partidas antigas que chegaram tarde ao banco.
  • 12 linhas por partida (verificar com count(DISTINCT match_id)).
  • A amostra da API pode não representar todos os jogadores (o site tem “Player Tracker” e
    “Prioritized Fetching”). É hipótese: registrar como limitação.
  • game_mode_version como proxy de patch: não documentado, não assumir.
 Privacidade: account_id identifica jogadores. Manter data/ fora do git e não publicar IDs no repositório nem no
 post.




Projeto Deadlock · Contexto: estabilidade da win rate entre dias                                          Página 3
