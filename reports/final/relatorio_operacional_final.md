# Relatório operacional final

## Resumo executivo

- Data de corte: **11/10/2026**
- Snapshot do MySQL: **3.160 registros**
- Modelo operacional: **MLP global**
- Critério de seleção: menor MAE no walk-forward
- Métricas de validação: MAE **29,3412**, RMSE **61,7443** e MAPE **29,4339%**
- Previsão total: **3.363 unidades**
- Recomendação total: **3.363 unidades**
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
| QUA | 14/10/2026 | Sol | não |  |
| QUI | 15/10/2026 | Sol | não |  |
| SAB_C | 17/10/2026 | Sol | não |  |
| SAB_E | 17/10/2026 | Sol | não |  |
| DOM_C | 18/10/2026 | Frio | não |  |
| DOM_E | 18/10/2026 | Frio | não |  |

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
| 13/10/2026 | terça-feira | QUA | 488 | 610 | 488 | 0 | 100,00% |
| 14/10/2026 | quarta-feira | QUI | 540 | 1.448 | 540 | 0 | 100,00% |
| 16/10/2026 | sexta-feira | SAB_C + SAB_E | 1.126 | 1.780 | 1.126 | 0 | 100,00% |
| 17/10/2026 | sábado | DOM_C + DOM_E | 1.209 | 1.883 | 1.209 | 0 | 100,00% |

## Recomendação por feira

| Venda | Feira | Previsão | Recomendação | Redução | Atendimento |
| --- | --- | --- | --- | --- | --- |
| 14/10/2026 | QUA | 488 | 488 | 0 | 100,00% |
| 15/10/2026 | QUI | 540 | 540 | 0 | 100,00% |
| 17/10/2026 | SAB_C | 529 | 529 | 0 | 100,00% |
| 17/10/2026 | SAB_E | 597 | 597 | 0 | 100,00% |
| 18/10/2026 | DOM_C | 797 | 797 | 0 | 100,00% |
| 18/10/2026 | DOM_E | 412 | 412 | 0 | 100,00% |

## Maiores ajustes por capacidade

_Nenhum registro._

O plano detalhado por produto está disponível em `plano_producao_operacional.csv`.

## Premissas, decisões e limitações

- **LIMITAÇÃO — Demanda possivelmente censurada:** 2361 registros (74.72%) podem representar vendas limitadas pela produção disponível.
- **LIMITAÇÃO — Capacidade provisória:** A capacidade usa o maior volume histórico por grupo e não representa uma medição física definitiva.
- **PREMISSA — Contexto futuro:** Clima e feriado foram informados explicitamente para seis feiras.
- **DECISÃO — Critério de seleção do modelo:** MLP global foi escolhido pelo menor MAE. O menor RMSE pertence a MLP global. O menor MAPE pertence a Média móvel (4 ocorrências).
- **PREMISSA — Validade do snapshot:** Modelo líder, previsões, proporções e capacidades devem ser recalculados após novas inserções no MySQL.
- **ALERTA — Revisar previsão: QUA | Comum:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: QUA | Doce:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: QUA | Especial:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: QUI | Comum:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: QUI | Doce:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: QUI | Especial:** previsão abaixo do mínimo histórico de vendas; atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: SAB_C | Comum:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: SAB_C | Doce:** previsão acima do máximo das últimas quatro ocorrências; atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: SAB_C | Especial:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: SAB_E | Comum:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: SAB_E | Doce:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: SAB_E | Especial:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: DOM_C | Comum:** previsão abaixo do mínimo histórico de vendas; atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: DOM_C | Doce:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: DOM_C | Especial:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: DOM_E | Comum:** previsão abaixo do mínimo histórico de vendas; atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: DOM_E | Doce:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.
- **ALERTA — Revisar previsão: DOM_E | Especial:** atributos futuros fora da faixa/categorias vistas no treino; nenhum modelo ou valor foi alterado.

## Validação da consolidação

Foram aprovadas **12 de 12** verificações da consolidação final.

## Decisão operacional

A recomendação apresentada utiliza o modelo líder revalidado no snapshot atual, distribui a previsão pelas participações históricas dos produtos e aplica o limite operacional por meio do dual Simplex. Uma nova execução deve ser realizada sempre que o banco de dados ou as premissas futuras forem atualizados.

## Auditoria diagnóstica das previsões

A comparação por série é diagnóstica: não escolhe outro modelo nem limita quantidades automaticamente. A faixa empírica de vendas não é um intervalo probabilístico, uma capacidade física ou uma prova de bug. Vendas e produção são registradas separadamente.

**Há séries que devem ser revisadas antes do uso operacional.**

| Série | Previsão | Máximo vendido | Média móvel 4 | Motivos |
| --- | --- | --- | --- | --- |
| QUA \| Comum | 440 | 502.0 | 446.25 | atributos futuros fora da faixa/categorias vistas no treino |
| QUA \| Doce | 30 | 36.0 | 31.75 | atributos futuros fora da faixa/categorias vistas no treino |
| QUA \| Especial | 18 | 22.0 | 17.25 | atributos futuros fora da faixa/categorias vistas no treino |
| QUI \| Comum | 510 | 1276.0 | 722.5 | atributos futuros fora da faixa/categorias vistas no treino |
| QUI \| Doce | 13 | 17.0 | 13.0 | atributos futuros fora da faixa/categorias vistas no treino |
| QUI \| Especial | 17 | 102.0 | 34.25 | previsão abaixo do mínimo histórico de vendas; atributos futuros fora da faixa/categorias vistas no treino |
| SAB_C \| Comum | 480 | 786.0 | 732.5 | atributos futuros fora da faixa/categorias vistas no treino |
| SAB_C \| Doce | 12 | 13.0 | 11.0 | previsão acima do máximo das últimas quatro ocorrências; atributos futuros fora da faixa/categorias vistas no treino |
| SAB_C \| Especial | 37 | 44.0 | 38.25 | atributos futuros fora da faixa/categorias vistas no treino |
| SAB_E \| Comum | 546 | 840.0 | 806.0 | atributos futuros fora da faixa/categorias vistas no treino |
| SAB_E \| Doce | 13 | 15.0 | 13.0 | atributos futuros fora da faixa/categorias vistas no treino |
| SAB_E \| Especial | 38 | 44.0 | 37.0 | atributos futuros fora da faixa/categorias vistas no treino |
| DOM_C \| Comum | 732 | 1062.0 | 995.75 | previsão abaixo do mínimo histórico de vendas; atributos futuros fora da faixa/categorias vistas no treino |
| DOM_C \| Doce | 16 | 18.0 | 14.75 | atributos futuros fora da faixa/categorias vistas no treino |
| DOM_C \| Especial | 49 | 71.0 | 60.25 | atributos futuros fora da faixa/categorias vistas no treino |
| DOM_E \| Comum | 372 | 682.0 | 616.0 | previsão abaixo do mínimo histórico de vendas; atributos futuros fora da faixa/categorias vistas no treino |
| DOM_E \| Doce | 7 | 13.0 | 7.25 | atributos futuros fora da faixa/categorias vistas no treino |
| DOM_E \| Especial | 33 | 47.0 | 35.0 | atributos futuros fora da faixa/categorias vistas no treino |

Evidências atuais e métricas por série: `reports/audit/`.
