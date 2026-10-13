# Relatório operacional final

## Resumo executivo

- Data de corte: **11/10/2026**
- Snapshot do MySQL: **3.160 registros**
- Modelo operacional: **MLP global**
- Critério de seleção: menor MAE no walk-forward
- Métricas de validação: MAE **29,3412**, RMSE **61,7443** e MAPE **29,4339%**
- Previsão total: **4.828 unidades**
- Recomendação total: **4.828 unidades**
- Redução por capacidade: **0 unidades**
- Fonte da capacidade: **historical_max** — configuração provisória
- Solver: **highs-ds**

## Ranking revalidado dos modelos

| Posição | Modelo | MAE | RMSE | MAPE (%) |
| --- | --- | --- | --- | --- |
| 1 | MLP global | 29,3412 | 61,7443 | 29,4339 |
| 2 | Média móvel (4 ocorrências) | 34,4444 | 66,7112 | 15,7230 |
| 3 | Suavização exponencial simples | 34,8368 | 67,8276 | 15,8252 |
| 4 | Média móvel (2 ocorrências) | 37,2722 | 71,7229 | 16,3582 |
| 5 | Média móvel (3 ocorrências) | 37,4926 | 70,9230 | 16,7144 |
| 6 | Naive (última ocorrência) | 38,9556 | 82,3707 | 19,8162 |
| 7 | Regressão linear global | 47,2472 | 66,7493 | 92,7839 |

## Contexto informado para as próximas feiras

| Feira | Venda | Clima | Feriado | Nome |
| --- | --- | --- | --- | --- |
| QUA | 14/10/2026 | SOL | não |  |
| QUI | 15/10/2026 | SOL | não |  |
| SAB_C | 17/10/2026 | SOL | não |  |
| SAB_E | 17/10/2026 | SOL | não |  |
| DOM_C | 18/10/2026 | SOL | não |  |
| DOM_E | 18/10/2026 | SOL | não |  |

## Capacidades utilizadas

| Produção | Feiras | Datas | Média | Capacidade | Data do máximo |
| --- | --- | --- | --- | --- | --- |
| terça-feira | QUA | 20 | 540,80 | 610 | 22/09/2026 |
| quarta-feira | QUI | 20 | 893,20 | 1.448 | 08/07/2026 |
| sexta-feira | SAB_C + SAB_E | 20 | 1.682,15 | 1.780 | 03/07/2026 |
| sábado | DOM_C + DOM_E | 20 | 1.747,15 | 1.883 | 19/09/2026 |

> As capacidades são máximos observados no histórico e não medições físicas definitivas.

## Recomendação por dia de produção

| Data | Dia | Feiras | Previsão | Capacidade | Recomendação | Redução | Atendimento |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 13/10/2026 | terça-feira | QUA | 566 | 610 | 566 | 0 | 100,00% |
| 14/10/2026 | quarta-feira | QUI | 827 | 1.448 | 827 | 0 | 100,00% |
| 16/10/2026 | sexta-feira | SAB_C + SAB_E | 1.668 | 1.780 | 1.668 | 0 | 100,00% |
| 17/10/2026 | sábado | DOM_C + DOM_E | 1.767 | 1.883 | 1.767 | 0 | 100,00% |

## Recomendação por feira

| Venda | Feira | Previsão | Recomendação | Redução | Atendimento |
| --- | --- | --- | --- | --- | --- |
| 14/10/2026 | QUA | 566 | 566 | 0 | 100,00% |
| 15/10/2026 | QUI | 827 | 827 | 0 | 100,00% |
| 17/10/2026 | SAB_C | 800 | 800 | 0 | 100,00% |
| 17/10/2026 | SAB_E | 868 | 868 | 0 | 100,00% |
| 18/10/2026 | DOM_C | 1.083 | 1.083 | 0 | 100,00% |
| 18/10/2026 | DOM_E | 684 | 684 | 0 | 100,00% |

## Maiores ajustes por capacidade

_Nenhum registro._

O plano detalhado por produto está disponível em `plano_producao_operacional.csv`.

## Premissas, decisões e limitações

- **LIMITAÇÃO — Demanda possivelmente censurada:** 2361 registros (74.72%) podem representar vendas limitadas pela produção disponível.
- **LIMITAÇÃO — Capacidade provisória:** A capacidade usa o maior volume histórico por grupo e não representa uma medição física definitiva.
- **PREMISSA — Contexto futuro:** Clima e feriado foram informados explicitamente para seis feiras.
- **DECISÃO — Critério de seleção do modelo:** MLP global foi escolhido pelo menor MAE. O menor RMSE pertence a MLP global. O menor MAPE pertence a Média móvel (4 ocorrências).
- **PREMISSA — Validade do snapshot:** Modelo líder, previsões, proporções e capacidades devem ser recalculados após novas inserções no MySQL.
- **ALERTA — Revisar previsão: QUA | Comum:** previsão acima do máximo histórico de vendas; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: QUA | Especial:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: QUI | Comum:** previsão acima do máximo das últimas quatro ocorrências; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: SAB_C | Comum:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: SAB_C | Doce:** previsão acima do máximo histórico de vendas; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: SAB_E | Comum:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: SAB_E | Doce:** previsão acima do máximo das últimas quatro ocorrências; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: DOM_C | Comum:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: DOM_C | Doce:** previsão acima do máximo das últimas quatro ocorrências; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: DOM_E | Comum:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.

## Validação da consolidação

Foram aprovadas **12 de 12** verificações da consolidação final.

## Decisão operacional

A recomendação apresentada utiliza o modelo líder revalidado no snapshot atual, distribui a previsão pelas participações históricas dos produtos e aplica o limite operacional por meio do dual Simplex. Uma nova execução deve ser realizada sempre que o banco de dados ou as premissas futuras forem atualizados.

## Auditoria diagnóstica das previsões

A comparação por série é diagnóstica: não escolhe outro modelo nem limita quantidades automaticamente. A faixa empírica de vendas não é um intervalo probabilístico, uma capacidade física ou uma prova de bug. Vendas e produção são registradas separadamente.

**Há séries que devem ser revisadas antes do uso operacional.**

| Série | Previsão | Máximo vendido | Média móvel 4 | Motivos |
| --- | --- | --- | --- | --- |
| QUA \| Comum | 516 | 502.0 | 446.25 | previsão acima do máximo histórico de vendas |
| QUA \| Especial | 19 | 22.0 | 17.25 | atributos futuros fora da faixa/categorias vistas no treino |
| QUI \| Comum | 777 | 1276.0 | 722.5 | previsão acima do máximo das últimas quatro ocorrências |
| SAB_C \| Comum | 748 | 786.0 | 732.5 | atributos futuros fora da faixa/categorias vistas no treino |
| SAB_C \| Doce | 13 | 13.0 | 11.0 | previsão acima do máximo histórico de vendas |
| SAB_E \| Comum | 813 | 840.0 | 806.0 | atributos futuros fora da faixa/categorias vistas no treino |
| SAB_E \| Doce | 15 | 15.0 | 13.0 | previsão acima do máximo das últimas quatro ocorrências |
| DOM_C \| Comum | 1009 | 1062.0 | 995.75 | atributos futuros fora da faixa/categorias vistas no treino |
| DOM_C \| Doce | 18 | 18.0 | 14.75 | previsão acima do máximo das últimas quatro ocorrências |
| DOM_E \| Comum | 640 | 682.0 | 616.0 | atributos futuros fora da faixa/categorias vistas no treino |

Evidências atuais e métricas por série: `reports/audit/`.
