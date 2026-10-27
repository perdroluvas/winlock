Semana 1: dados de Deadlock, estatística e
primeiro gráfico
Plano de estudo e projeto para a primeira semana do foco em dados (Python, estatística e ML).


Objetivo da semana
Ao final, você terá um repositório no GitHub com um pipeline que baixa os dados, uma análise
exploratória e um gráfico para o primeiro post. A pergunta que guia tudo:

   A taxa de vitória de um herói é confiável, ou é ruído estatístico?

Ela é simples e usa estatística de verdade (intervalos de confiança). Também rende um post com
ângulo próprio, em vez de só mais um ranking de heróis.


Sobre a fonte de dados
A Deadlock API é de código aberto, gratuita, e oferece uma API de dados de jogo (partidas,
jogadores, estatísticas) e também snapshots do banco para análise offline. A documentação
interativa fica em api.deadlock-api.com/docs. Como é um projeto comunitário, confira os
endpoints, os limites de requisição e os termos de uso antes de começar, porque eles mudam.
Para a primeira semana, prefira baixar uma amostra (ou um dump pequeno) em vez de fazer
milhares de requisições.


Plano dia a dia (1 a 2h por dia)
Dia 1: preparar o ambiente
 • Crie o repositório com esta estrutura: data/ (ignorado no git), src/, notebooks/,
   README.md, requirements.txt.
 • Crie um ambiente virtual e instale pandas, numpy, matplotlib, seaborn, scipy, statsmodels,
   requests.
 • Leia a documentação da API e anote os endpoints de partidas e heróis.
 • Faça o primeiro commit, mesmo só com o README e o objetivo.


Dia 2: coletar os dados
 • Escreva um script em src/ (não em notebook) que baixa uma amostra de partidas, com
   pausas entre requisições, e salva em Parquet ou CSV.
 • Baixe também a tabela de heróis (IDs e nomes) pela Assets API.
 • Meta: algumas dezenas de milhares de linhas de partidas, o suficiente para tirar
   conclusões.


Dia 3: entender e limpar
 • Veja df.info(), df.describe() e valores nulos.
 • Cheque duplicatas, partidas incompletas e outliers (por exemplo, duração absurda).
 • Anote no README o que você descartou e por quê. Isso é parte do portfólio.
Dia 4: estatística básica (estudo + prática)
 • Estude: média vs. mediana, desvio padrão, distribuição binomial, intervalo de confiança para
   proporções.
 • Aplique no seu dataset: calcule a taxa de vitória por herói, o número de partidas por herói e o
   intervalo de confiança de Wilson
   (statsmodels.stats.proportion.proportion_confint com method="wilson").


Dia 5: análise exploratória
 • Distribuição da duração das partidas.
 • Taxa de vitória por herói com intervalo de confiança.
 • Relação entre popularidade e taxa de vitória.
 • Um teste de hipótese simples, por exemplo se a taxa de vitória de um herói difere de 50% de
   forma estatisticamente significativa (teste binomial).


Dia 6: o gráfico do post
 • Faça um gráfico horizontal de taxa de vitória por herói com barras de erro (IC 95%),
   ordenado por taxa e com a linha de 50% destacada.
 • O ponto do post: heróis com poucas partidas têm intervalos largos, então um "herói com 55%
   de vitória" pode não significar nada. Isso mostra que você entende incerteza, o que muita
   gente em dados ignora.
 • Cuide da estética: título claro, eixos legíveis, cores consistentes.


Dia 7: fechar e publicar
 • Organize o notebook, escreva o README (pergunta, dados, método, resultado, limitações) e
   faça o push.
 • Rascunhe o post e publique se estiver confortável. Se não, deixe para a semana 2.


Estrutura sugerida para o post
 1. Gancho: uma frase sobre o que parece óbvio ("o herói X tem a maior taxa de vitória") e por
   que pode ser enganoso.
 2. O gráfico.
 3. Duas ou três frases sobre o método (amostra, intervalo de confiança).
 4. Uma limitação honesta (por exemplo, o viés da amostra: quem tem dados na API pode não
   representar todos os jogadores).
 5. Link do repositório.


Material de estudo para a semana
 • Um curso ou livro introdutório de estatística aplicada com Python. Escolha um só e use como
   consulta, não para maratonar.
 • Documentação do Pandas (seção de groupby e merge).
 • Comece pelo conceito de intervalo de confiança para proporções. É curto e já dá resultado no
   projeto.


Armadilhas comuns
 • Querer baixar tudo: comece pequeno. Muitos dados atrapalham mais que ajudam na
   semana 1.
 • Fazer tudo no notebook: o script de coleta em src/ já prepara o terreno para o pipeline
   reproduzível das próximas semanas.
 • Perfeccionismo: um gráfico bom e publicado vale mais que um perfeito na gaveta.
 • Dados desatualizados: o jogo recebe patches que mudam o equilíbrio. Registre o período ou
   a versão dos dados no README.


Próximo passo
Na semana 2, você pode comparar patches ou faixas de habilidade e começar a montar variáveis
para o modelo de previsão de partidas.
