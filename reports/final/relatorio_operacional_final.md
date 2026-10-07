# Relatório operacional final

## Resumo executivo

- Data de corte: **04/10/2026**
- Snapshot do MySQL: **3.002 registros**
- Modelo operacional: **MLP global**
- Critério de seleção: menor MAE no walk-forward
- Métricas de validação: MAE **36,3347**, RMSE **79,5986** e MAPE **29,5914%**
- Previsão total: **4.825 unidades**
- Recomendação total: **4.772 unidades**
- Redução por capacidade: **53 unidades**
- Fonte da capacidade: **historical_max** — configuração provisória
- Solver: **highs-ds**

## Ranking revalidado dos modelos

| Posição | Modelo | MAE | RMSE | MAPE (%) |
| --- | --- | --- | --- | --- |
| 1 | MLP global | 36,3347 | 79,5986 | 29,5914 |
| 2 | Média móvel (4 ocorrências) | 38,6250 | 74,0871 | 16,8128 |
| 3 | Suavização exponencial simples | 39,5394 | 75,9570 | 16,8089 |
| 4 | Média móvel (3 ocorrências) | 41,5185 | 77,4082 | 17,4059 |
| 5 | Média móvel (2 ocorrências) | 41,7833 | 79,8557 | 17,3055 |
| 6 | Naive (última ocorrência) | 44,2333 | 89,5118 | 19,9933 |
| 7 | Regressão linear global | 46,3618 | 67,2343 | 82,2255 |

## Contexto informado para as próximas feiras

| Feira | Venda | Clima | Feriado | Nome |
| --- | --- | --- | --- | --- |
| QUA | 07/10/2026 | Sol | não |  |
| QUI | 08/10/2026 | Sol | não |  |
| SAB_C | 10/10/2026 | Sol | não |  |
| SAB_E | 10/10/2026 | Sol | não |  |
| DOM_C | 11/10/2026 | Sol | não |  |
| DOM_E | 11/10/2026 | Sol | não |  |

## Capacidades utilizadas

| Produção | Feiras | Datas | Média | Capacidade | Data do máximo |
| --- | --- | --- | --- | --- | --- |
| terça-feira | QUA | 19 | 537,84 | 610 | 22/09/2026 |
| quarta-feira | QUI | 19 | 895,00 | 1.448 | 08/07/2026 |
| sexta-feira | SAB_C + SAB_E | 19 | 1.678,89 | 1.780 | 03/07/2026 |
| sábado | DOM_C + DOM_E | 19 | 1.752,11 | 1.883 | 19/09/2026 |

> As capacidades são máximos observados no histórico e não medições físicas definitivas.

## Recomendação por dia de produção

| Data | Dia | Feiras | Previsão | Capacidade | Recomendação | Redução | Atendimento |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 06/10/2026 | terça-feira | QUA | 464 | 610 | 464 | 0 | 100,00% |
| 07/10/2026 | quarta-feira | QUI | 727 | 1.448 | 727 | 0 | 100,00% |
| 09/10/2026 | sexta-feira | SAB_C + SAB_E | 1.698 | 1.780 | 1.698 | 0 | 100,00% |
| 10/10/2026 | sábado | DOM_C + DOM_E | 1.936 | 1.883 | 1.883 | 53 | 97,26% |

## Recomendação por feira

| Venda | Feira | Previsão | Recomendação | Redução | Atendimento |
| --- | --- | --- | --- | --- | --- |
| 07/10/2026 | QUA | 464 | 464 | 0 | 100,00% |
| 08/10/2026 | QUI | 727 | 727 | 0 | 100,00% |
| 10/10/2026 | SAB_C | 884 | 884 | 0 | 100,00% |
| 10/10/2026 | SAB_E | 814 | 814 | 0 | 100,00% |
| 11/10/2026 | DOM_C | 1.211 | 1.178 | 33 | 97,27% |
| 11/10/2026 | DOM_E | 725 | 705 | 20 | 97,24% |

## Maiores ajustes por capacidade

| Feira | Categoria | Produto | Previsão | Recomendação | Redução |
| --- | --- | --- | --- | --- | --- |
| DOM_C | Comum | Frango catupiri | 230 | 224 | 6 |
| DOM_C | Comum | Carne com queijo | 189 | 184 | 5 |
| DOM_C | Comum | Carne | 129 | 125 | 4 |
| DOM_E | Comum | Frango catupiri | 141 | 137 | 4 |
| DOM_E | Comum | Carne com queijo | 113 | 110 | 3 |
| DOM_C | Comum | Camarão | 59 | 57 | 2 |
| DOM_C | Comum | Carne seca | 62 | 60 | 2 |
| DOM_C | Especial | Especial de carne | 66 | 64 | 2 |
| DOM_C | Comum | Pizza | 63 | 61 | 2 |
| DOM_C | Comum | Queijo | 79 | 77 | 2 |
| DOM_E | Comum | Carne | 72 | 70 | 2 |
| DOM_C | Comum | 4 queijos | 20 | 19 | 1 |
| DOM_C | Comum | Bauru | 20 | 19 | 1 |
| DOM_C | Comum | Caipira | 25 | 24 | 1 |
| DOM_C | Comum | Calabresa com queijo | 29 | 28 | 1 |

O plano detalhado por produto está disponível em `plano_producao_operacional.csv`.

## Premissas, decisões e limitações

- **LIMITAÇÃO — Demanda possivelmente censurada:** 2241 registros (74.65%) podem representar vendas limitadas pela produção disponível.
- **LIMITAÇÃO — Capacidade provisória:** A capacidade usa o maior volume histórico por grupo e não representa uma medição física definitiva.
- **PREMISSA — Contexto futuro:** Clima e feriado foram informados explicitamente para seis feiras.
- **DECISÃO — Critério de seleção do modelo:** MLP global foi escolhido pelo menor MAE. O menor RMSE pertence a Regressão linear global. O menor MAPE pertence a Suavização exponencial simples.
- **PREMISSA — Validade do snapshot:** Modelo líder, previsões, proporções e capacidades devem ser recalculados após novas inserções no MySQL.

## Validação da consolidação

Foram aprovadas **12 de 12** verificações da consolidação final.

## Decisão operacional

A recomendação apresentada utiliza o modelo líder revalidado no snapshot atual, distribui a previsão pelas participações históricas dos produtos e aplica o limite operacional por meio do dual Simplex. Uma nova execução deve ser realizada sempre que o banco de dados ou as premissas futuras forem atualizados.
