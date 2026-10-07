"""Consolidação operacional e documentação final — Fase 14.

Este módulo não recalcula previsões nem altera a solução do Simplex. Ele reúne
as decisões e saídas já validadas das fases anteriores em um pacote operacional
composto por um resumo da execução, um plano de produção simplificado, uma
relação explícita de alertas e limitações e um relatório em Markdown.
"""

from __future__ import annotations

from math import isfinite
from pathlib import Path
from typing import TypeAlias

import pandas as pd


FinalArtifact: TypeAlias = pd.DataFrame | str

FAIR_ORDER = ["QUA", "QUI", "SAB_C", "SAB_E", "DOM_C", "DOM_E"]
CATEGORY_ORDER = ["Comum", "Especial", "Doce"]

SUMMARY_KEY = "resumo_execucao_final"
PLAN_KEY = "plano_producao_operacional"
ALERTS_KEY = "alertas_limitacoes_operacionais"
VALIDATION_KEY = "validacao_consolidacao_final"
REPORT_KEY = "relatorio_operacional_final"


def _require_dataframe(
    analyses: dict[str, pd.DataFrame],
    key: str,
) -> pd.DataFrame:
    """Obtém um DataFrame obrigatório e rejeita estruturas incompletas."""

    dataframe = analyses.get(key)
    if not isinstance(dataframe, pd.DataFrame) or dataframe.empty:
        raise ValueError(f"Resultado obrigatório ausente ou vazio: {key}.")
    return dataframe.copy()


def _format_number(value: int | float, decimal_places: int = 0) -> str:
    """Formata números segundo a convenção brasileira para o relatório."""

    formatted = f"{float(value):,.{decimal_places}f}"
    return formatted.replace(",", "X").replace(".", ",").replace("X", ".")


def _format_date(value: object) -> str:
    """Formata uma data para apresentação e preserva valores vazios."""

    if pd.isna(value):
        return ""
    return pd.Timestamp(value).strftime("%d/%m/%Y")


def _markdown_table(
    dataframe: pd.DataFrame,
    columns: list[str],
    labels: list[str],
) -> str:
    """Renderiza uma tabela Markdown sem dependência do pacote tabulate."""

    if dataframe.empty:
        return "_Nenhum registro._"

    rows = [
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join(["---"] * len(labels)) + " |",
    ]
    for _, row in dataframe[columns].iterrows():
        values = []
        for column in columns:
            value = row[column]
            text = "" if pd.isna(value) else str(value)
            values.append(text.replace("|", "\\|").replace("\n", " "))
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def create_operational_plan(
    recommendations: pd.DataFrame,
) -> pd.DataFrame:
    """Cria uma visão enxuta somente com itens que precisam ser produzidos."""

    required_columns = [
        "data_producao_prevista",
        "data_venda_prevista",
        "dia_producao",
        "feira",
        "categoria",
        "id_produto",
        "produto",
        "produto_permitido",
        "previsao_produto",
        "recomendacao_producao",
        "reducao_em_relacao_previsao",
        "percentual_atendimento_produto",
    ]
    missing = [
        column for column in required_columns if column not in recommendations
    ]
    if missing:
        raise ValueError(
            "Colunas ausentes nas recomendações finais: " + ", ".join(missing)
        )

    plan = recommendations.loc[
        recommendations["produto_permitido"].eq(1)
        & recommendations["recomendacao_producao"].gt(0),
        required_columns,
    ].copy()
    plan["observacao_operacional"] = "produzir conforme recomendação"
    reduced = plan["reducao_em_relacao_previsao"].gt(0)
    plan.loc[reduced, "observacao_operacional"] = (
        "quantidade reduzida por limite de capacidade"
    )

    fair_rank = {fair: index for index, fair in enumerate(FAIR_ORDER)}
    category_rank = {
        category: index for index, category in enumerate(CATEGORY_ORDER)
    }
    plan["_ordem_feira"] = plan["feira"].map(fair_rank)
    plan["_ordem_categoria"] = plan["categoria"].map(category_rank)

    return (
        plan.sort_values(
            [
                "data_producao_prevista",
                "_ordem_feira",
                "_ordem_categoria",
                "id_produto",
            ]
        )
        .drop(columns=["_ordem_feira", "_ordem_categoria"])
        .reset_index(drop=True)
    )


def create_final_summary(
    operational_decision: pd.Series,
    optimization_decision: pd.Series,
    category_forecasts: pd.DataFrame,
    recommendations: pd.DataFrame,
    plan: pd.DataFrame,
    censored_count: int,
    operational_validation: pd.DataFrame,
    distribution_validation: pd.DataFrame,
    optimization_validation: pd.DataFrame,
) -> pd.DataFrame:
    """Resume em uma linha o snapshot, o modelo e a recomendação final."""

    source_records = int(optimization_decision["registros_origem_mysql"])
    censored_percentage = (
        float(censored_count) / source_records * 100 if source_records else 0.0
    )
    validations = [
        operational_validation,
        distribution_validation,
        optimization_validation,
    ]
    validation_total = sum(len(item) for item in validations)
    validation_ok = sum(
        int(item["resultado"].eq("OK").sum()) for item in validations
    )

    return pd.DataFrame(
        [
            {
                "data_corte": pd.Timestamp(optimization_decision["data_corte"]),
                "registros_origem_mysql": source_records,
                "possiveis_casos_demanda_censurada": int(censored_count),
                "percentual_possivel_demanda_censurada": round(
                    censored_percentage, 4
                ),
                "modelo_operacional": operational_decision[
                    "modelo_operacional"
                ],
                "nome_modelo_operacional": operational_decision[
                    "nome_modelo_operacional"
                ],
                "criterio_selecao": operational_decision[
                    "criterio_selecao"
                ],
                "mae_validacao": float(
                    operational_decision["mae_validacao"]
                ),
                "rmse_validacao": float(
                    operational_decision["rmse_validacao"]
                ),
                "mape_validacao": float(
                    operational_decision["mape_validacao"]
                ),
                "configuracao_modelo": operational_decision[
                    "configuracao_operacional_selecionada"
                ],
                "linhas_treino_modelo_final": int(
                    operational_decision["linhas_treino_modelo_final"]
                ),
                "series_categoria_feira_previstas": len(category_forecasts),
                "combinacoes_produto_feira_auditadas": len(recommendations),
                "itens_com_producao_recomendada": len(plan),
                "fonte_capacidade": optimization_decision[
                    "fonte_capacidade"
                ],
                "algoritmo_solver": optimization_decision[
                    "algoritmo_solver"
                ],
                "grupos_com_capacidade_ativa": int(
                    optimization_decision[
                        "grupos_com_capacidade_ativa"
                    ]
                ),
                "previsao_total": int(
                    optimization_decision["previsao_total_produtos"]
                ),
                "recomendacao_total": int(
                    optimization_decision[
                        "recomendacao_total_producao"
                    ]
                ),
                "reducao_total": int(
                    optimization_decision[
                        "reducao_total_por_capacidade"
                    ]
                ),
                "validacoes_anteriores_aprovadas": validation_ok,
                "validacoes_anteriores_totais": validation_total,
                "status": (
                    "Pipeline completo consolidado para uso operacional."
                ),
            }
        ]
    )


def create_alerts_and_limitations(
    summary: pd.DataFrame,
    ranking: pd.DataFrame,
    future_context: pd.DataFrame,
) -> pd.DataFrame:
    """Registra premissas que precisam acompanhar a recomendação."""

    row = summary.iloc[0]
    leader = ranking.sort_values("posicao").iloc[0]
    best_rmse = ranking.sort_values("rmse").iloc[0]
    valid_mape = ranking.dropna(subset=["mape"])
    best_mape = (
        valid_mape.sort_values("mape").iloc[0] if not valid_mape.empty else None
    )

    alerts = [
        {
            "nivel": "LIMITAÇÃO",
            "item": "Demanda possivelmente censurada",
            "detalhes": (
                f"{int(row['possiveis_casos_demanda_censurada'])} registros "
                f"({float(row['percentual_possivel_demanda_censurada']):.2f}%) "
                "podem representar vendas limitadas pela produção disponível."
            ),
        },
        {
            "nivel": "LIMITAÇÃO",
            "item": "Capacidade provisória",
            "detalhes": (
                "A capacidade usa o maior volume histórico por grupo e não "
                "representa uma medição física definitiva."
            ),
        },
        {
            "nivel": "PREMISSA",
            "item": "Contexto futuro",
            "detalhes": (
                "Clima e feriado foram informados explicitamente para seis "
                "feiras."
                if not future_context.empty
                else "O modelo líder não exigiu clima ou feriado futuro."
            ),
        },
        {
            "nivel": "DECISÃO",
            "item": "Critério de seleção do modelo",
            "detalhes": (
                f"{leader['nome_modelo']} foi escolhido pelo menor MAE. "
                f"O menor RMSE pertence a {best_rmse['nome_modelo']}."
                + (
                    f" O menor MAPE pertence a {best_mape['nome_modelo']}."
                    if best_mape is not None
                    else ""
                )
            ),
        },
        {
            "nivel": "PREMISSA",
            "item": "Validade do snapshot",
            "detalhes": (
                "Modelo líder, previsões, proporções e capacidades devem ser "
                "recalculados após novas inserções no MySQL."
            ),
        },
    ]
    return pd.DataFrame(alerts)


def validate_final_consolidation(
    ranking: pd.DataFrame,
    operational_decision: pd.Series,
    optimization_decision: pd.Series,
    category_forecasts: pd.DataFrame,
    recommendations: pd.DataFrame,
    day_summary: pd.DataFrame,
    fair_summary: pd.DataFrame,
    future_context: pd.DataFrame,
    operational_validation: pd.DataFrame,
    distribution_validation: pd.DataFrame,
    optimization_validation: pd.DataFrame,
    summary: pd.DataFrame,
    plan: pd.DataFrame,
    alerts: pd.DataFrame,
) -> pd.DataFrame:
    """Audita a consolidação sem refazer decisões das fases anteriores."""

    checks: list[dict[str, str]] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "teste": name,
                "resultado": "OK" if passed else "FALHA",
                "detalhes": details,
            }
        )

    upstream = {
        "previsão operacional": operational_validation,
        "distribuição por produto": distribution_validation,
        "recomendação por Simplex": optimization_validation,
    }
    failed_upstream = {
        name: int(frame["resultado"].ne("OK").sum())
        for name, frame in upstream.items()
    }
    add_check(
        "Validações anteriores preservadas",
        all(value == 0 for value in failed_upstream.values()),
        f"Falhas encontradas: {failed_upstream}",
    )

    ranking_leader = ranking.sort_values("posicao").iloc[0]
    operational_model = str(operational_decision["modelo_operacional"])
    add_check(
        "Modelo operacional igual ao líder revalidado",
        str(ranking_leader["modelo"]) == operational_model,
        (
            f"Ranking: {ranking_leader['modelo']}; operacional: "
            f"{operational_model}"
        ),
    )

    snapshots = {
        int(operational_decision["registros_origem_mysql"]),
        int(optimization_decision["registros_origem_mysql"]),
        int(summary.iloc[0]["registros_origem_mysql"]),
    }
    cutoff_dates = {
        pd.Timestamp(operational_decision["data_corte"]).normalize(),
        pd.Timestamp(optimization_decision["data_corte"]).normalize(),
        pd.Timestamp(summary.iloc[0]["data_corte"]).normalize(),
    }
    add_check(
        "Snapshot e data de corte consistentes",
        len(snapshots) == 1 and len(cutoff_dates) == 1,
        f"Registros: {sorted(snapshots)}; datas: {sorted(cutoff_dates)}",
    )

    coverage_ok = (
        len(category_forecasts) == 18
        and category_forecasts["feira"].nunique() == 6
        and len(recommendations) == 162
        and recommendations["feira"].nunique() == 6
        and recommendations["id_produto"].nunique() == 27
    )
    add_check(
        "Cobertura operacional completa",
        coverage_ok,
        (
            f"Categorias: {len(category_forecasts)}; combinações produto + "
            f"feira: {len(recommendations)}"
        ),
    )

    plan_valid = (
        not plan.empty
        and plan["produto_permitido"].eq(1).all()
        and plan["recomendacao_producao"].gt(0).all()
    )
    add_check(
        "Plano contém somente produção permitida e positiva",
        bool(plan_valid),
        f"Linhas acionáveis: {len(plan)}",
    )

    duplicated = int(
        plan.duplicated(
            subset=["data_venda_prevista", "feira", "id_produto"]
        ).sum()
    )
    add_check(
        "Plano operacional sem duplicidades",
        duplicated == 0,
        f"Duplicidades encontradas: {duplicated}",
    )

    total_recommendations = int(recommendations["recomendacao_producao"].sum())
    plan_total = int(plan["recomendacao_producao"].sum())
    summary_total = int(summary.iloc[0]["recomendacao_total"])
    add_check(
        "Totais do plano reconciliados",
        total_recommendations == plan_total == summary_total,
        (
            f"Recomendações: {total_recommendations}; plano: {plan_total}; "
            f"resumo: {summary_total}"
        ),
    )

    day_total = int(day_summary["recomendacao_total"].sum())
    fair_total = int(fair_summary["recomendacao_total"].sum())
    add_check(
        "Resumos por dia e feira reconciliados",
        day_total == fair_total == summary_total,
        f"Dias: {day_total}; feiras: {fair_total}; final: {summary_total}",
    )

    numeric_values = summary[
        [
            "mae_validacao",
            "rmse_validacao",
            "mape_validacao",
            "previsao_total",
            "recomendacao_total",
            "reducao_total",
        ]
    ].stack()
    finite_nonnegative = all(
        isfinite(float(value)) and float(value) >= 0 for value in numeric_values
    )
    add_check(
        "Indicadores finais válidos",
        finite_nonnegative,
        "Métricas e totais devem ser finitos e não negativos.",
    )

    context_required = str(
        operational_decision["contexto_futuro_requerido"]
    ).lower() == "sim"
    context_ok = (
        len(future_context) == 6
        and set(future_context["feira"]) == set(FAIR_ORDER)
        if context_required
        else future_context.empty
    )
    add_check(
        "Contexto futuro documentado",
        bool(context_ok),
        (
            f"Contexto requerido: {'sim' if context_required else 'não'}; "
            f"linhas: {len(future_context)}"
        ),
    )

    required_alerts = {
        "Demanda possivelmente censurada",
        "Capacidade provisória",
        "Contexto futuro",
        "Critério de seleção do modelo",
        "Validade do snapshot",
    }
    add_check(
        "Premissas e limitações documentadas",
        required_alerts.issubset(set(alerts["item"])),
        f"Itens registrados: {sorted(set(alerts['item']))}",
    )

    reduction_expected = int(
        optimization_decision["previsao_total_produtos"]
        - optimization_decision["recomendacao_total_producao"]
    )
    add_check(
        "Redução final explicada pela capacidade",
        reduction_expected
        == int(optimization_decision["reducao_total_por_capacidade"]),
        f"Redução reconciliada: {reduction_expected}",
    )

    return pd.DataFrame(checks)


def render_operational_report(
    ranking: pd.DataFrame,
    summary: pd.DataFrame,
    future_context: pd.DataFrame,
    capacities: pd.DataFrame,
    day_summary: pd.DataFrame,
    fair_summary: pd.DataFrame,
    recommendations: pd.DataFrame,
    alerts: pd.DataFrame,
    validation: pd.DataFrame,
) -> str:
    """Produz o relatório final legível em Markdown."""

    row = summary.iloc[0]
    ranking_display = ranking[
        ["posicao", "nome_modelo", "mae", "rmse", "mape"]
    ].copy()
    for metric in ["mae", "rmse", "mape"]:
        ranking_display[metric] = ranking_display[metric].map(
            lambda value: _format_number(value, 4)
        )

    context_display = future_context.copy()
    if not context_display.empty:
        context_display["data_venda_prevista"] = context_display[
            "data_venda_prevista"
        ].map(_format_date)
        context_display["feriado"] = context_display["eh_feriado"].map(
            {0: "não", 1: "sim"}
        )

    capacity_display = capacities.copy()
    capacity_display["producao_media_historica"] = capacity_display[
        "producao_media_historica"
    ].map(lambda value: _format_number(value, 2))
    capacity_display["capacidade_estimada"] = capacity_display[
        "capacidade_estimada"
    ].map(lambda value: _format_number(value))
    capacity_display["data_producao_maxima"] = capacity_display[
        "data_producao_maxima"
    ].map(_format_date)

    day_display = day_summary.copy()
    day_display["data_producao_prevista"] = day_display[
        "data_producao_prevista"
    ].map(_format_date)
    for column in [
        "previsao_total",
        "capacidade_estimada",
        "recomendacao_total",
        "reducao_total",
    ]:
        day_display[column] = day_display[column].map(_format_number)
    day_display["percentual_atendimento"] = day_display[
        "percentual_atendimento"
    ].map(lambda value: _format_number(value, 2) + "%")

    fair_display = fair_summary.copy()
    fair_display["data_venda_prevista"] = fair_display[
        "data_venda_prevista"
    ].map(_format_date)
    for column in ["previsao_total", "recomendacao_total", "reducao_total"]:
        fair_display[column] = fair_display[column].map(_format_number)
    fair_display["percentual_atendimento"] = fair_display[
        "percentual_atendimento"
    ].map(lambda value: _format_number(value, 2) + "%")

    reductions = recommendations.loc[
        recommendations["reducao_em_relacao_previsao"].gt(0),
        [
            "feira",
            "categoria",
            "produto",
            "previsao_produto",
            "recomendacao_producao",
            "reducao_em_relacao_previsao",
        ],
    ].sort_values(
        ["reducao_em_relacao_previsao", "feira", "produto"],
        ascending=[False, True, True],
    ).head(15)

    alert_lines = "\n".join(
        f"- **{item['nivel']} — {item['item']}:** {item['detalhes']}"
        for _, item in alerts.iterrows()
    )
    validation_ok = int(validation["resultado"].eq("OK").sum())

    sections = [
        "# Relatório operacional final",
        "",
        "## Resumo executivo",
        "",
        f"- Data de corte: **{_format_date(row['data_corte'])}**",
        (
            "- Snapshot do MySQL: "
            f"**{_format_number(row['registros_origem_mysql'])} registros**"
        ),
        (
            "- Modelo operacional: "
            f"**{row['nome_modelo_operacional']}**"
        ),
        f"- Critério de seleção: {row['criterio_selecao']}",
        (
            "- Métricas de validação: MAE "
            f"**{_format_number(row['mae_validacao'], 4)}**, RMSE "
            f"**{_format_number(row['rmse_validacao'], 4)}** e MAPE "
            f"**{_format_number(row['mape_validacao'], 4)}%**"
        ),
        (
            "- Previsão total: "
            f"**{_format_number(row['previsao_total'])} unidades**"
        ),
        (
            "- Recomendação total: "
            f"**{_format_number(row['recomendacao_total'])} unidades**"
        ),
        (
            "- Redução por capacidade: "
            f"**{_format_number(row['reducao_total'])} unidades**"
        ),
        (
            "- Fonte da capacidade: "
            f"**{row['fonte_capacidade']}** — configuração provisória"
        ),
        f"- Solver: **{row['algoritmo_solver']}**",
        "",
        "## Ranking revalidado dos modelos",
        "",
        _markdown_table(
            ranking_display,
            ["posicao", "nome_modelo", "mae", "rmse", "mape"],
            ["Posição", "Modelo", "MAE", "RMSE", "MAPE (%)"],
        ),
        "",
        "## Contexto informado para as próximas feiras",
        "",
        (
            _markdown_table(
                context_display,
                [
                    "feira",
                    "data_venda_prevista",
                    "clima",
                    "feriado",
                    "nome_feriado",
                ],
                ["Feira", "Venda", "Clima", "Feriado", "Nome"],
            )
            if not context_display.empty
            else "_O modelo líder não exigiu contexto futuro._"
        ),
        "",
        "## Capacidades utilizadas",
        "",
        _markdown_table(
            capacity_display,
            [
                "dia_producao",
                "feiras_atendidas",
                "quantidade_datas_historicas",
                "producao_media_historica",
                "capacidade_estimada",
                "data_producao_maxima",
            ],
            [
                "Produção",
                "Feiras",
                "Datas",
                "Média",
                "Capacidade",
                "Data do máximo",
            ],
        ),
        "",
        (
            "> As capacidades são máximos observados no histórico e não "
            "medições físicas definitivas."
        ),
        "",
        "## Recomendação por dia de produção",
        "",
        _markdown_table(
            day_display,
            [
                "data_producao_prevista",
                "dia_producao",
                "feiras_atendidas",
                "previsao_total",
                "capacidade_estimada",
                "recomendacao_total",
                "reducao_total",
                "percentual_atendimento",
            ],
            [
                "Data",
                "Dia",
                "Feiras",
                "Previsão",
                "Capacidade",
                "Recomendação",
                "Redução",
                "Atendimento",
            ],
        ),
        "",
        "## Recomendação por feira",
        "",
        _markdown_table(
            fair_display,
            [
                "data_venda_prevista",
                "feira",
                "previsao_total",
                "recomendacao_total",
                "reducao_total",
                "percentual_atendimento",
            ],
            [
                "Venda",
                "Feira",
                "Previsão",
                "Recomendação",
                "Redução",
                "Atendimento",
            ],
        ),
        "",
        "## Maiores ajustes por capacidade",
        "",
        _markdown_table(
            reductions,
            [
                "feira",
                "categoria",
                "produto",
                "previsao_produto",
                "recomendacao_producao",
                "reducao_em_relacao_previsao",
            ],
            [
                "Feira",
                "Categoria",
                "Produto",
                "Previsão",
                "Recomendação",
                "Redução",
            ],
        ),
        "",
        (
            "O plano detalhado por produto está disponível em "
            "`plano_producao_operacional.csv`."
        ),
        "",
        "## Premissas, decisões e limitações",
        "",
        alert_lines,
        "",
        "## Validação da consolidação",
        "",
        (
            f"Foram aprovadas **{validation_ok} de {len(validation)}** "
            "verificações da consolidação final."
        ),
        "",
        "## Decisão operacional",
        "",
        (
            "A recomendação apresentada utiliza o modelo líder revalidado no "
            "snapshot atual, distribui a previsão pelas participações "
            "históricas dos produtos e aplica o limite operacional por meio "
            "do dual Simplex. Uma nova execução deve ser realizada sempre que "
            "o banco de dados ou as premissas futuras forem atualizados."
        ),
        "",
    ]
    return "\n".join(sections)


def run_final_consolidation(
    ranking: pd.DataFrame,
    operational_analyses: dict[str, pd.DataFrame],
    distribution_analyses: dict[str, pd.DataFrame],
    recommendation_analyses: dict[str, pd.DataFrame],
    censored_count: int,
) -> dict[str, FinalArtifact]:
    """Consolida as saídas validadas do pipeline completo."""

    if ranking.empty:
        raise ValueError("O ranking de modelos está vazio.")

    category_forecasts = _require_dataframe(
        operational_analyses, "previsoes_operacionais_categoria_feira"
    )
    operational_validation = _require_dataframe(
        operational_analyses, "validacao_previsao_operacional"
    )
    operational_decision = _require_dataframe(
        operational_analyses, "decisao_modelo_operacional"
    ).iloc[0]
    future_context = operational_analyses.get(
        "contexto_futuro_previsao", pd.DataFrame()
    ).copy()

    distribution_validation = _require_dataframe(
        distribution_analyses, "validacao_distribuicao_produtos"
    )
    recommendations = _require_dataframe(
        recommendation_analyses,
        "recomendacoes_producao_produto_feira",
    )
    capacities = _require_dataframe(
        recommendation_analyses, "capacidades_estimadas_producao"
    )
    day_summary = _require_dataframe(
        recommendation_analyses, "resumo_recomendacao_por_dia"
    )
    fair_summary = _require_dataframe(
        recommendation_analyses, "resumo_recomendacao_por_feira"
    )
    optimization_validation = _require_dataframe(
        recommendation_analyses, "validacao_recomendacao_producao"
    )
    optimization_decision = _require_dataframe(
        recommendation_analyses, "decisao_recomendacao_producao"
    ).iloc[0]

    plan = create_operational_plan(recommendations)
    summary = create_final_summary(
        operational_decision=operational_decision,
        optimization_decision=optimization_decision,
        category_forecasts=category_forecasts,
        recommendations=recommendations,
        plan=plan,
        censored_count=censored_count,
        operational_validation=operational_validation,
        distribution_validation=distribution_validation,
        optimization_validation=optimization_validation,
    )
    alerts = create_alerts_and_limitations(
        summary=summary,
        ranking=ranking,
        future_context=future_context,
    )
    validation = validate_final_consolidation(
        ranking=ranking,
        operational_decision=operational_decision,
        optimization_decision=optimization_decision,
        category_forecasts=category_forecasts,
        recommendations=recommendations,
        day_summary=day_summary,
        fair_summary=fair_summary,
        future_context=future_context,
        operational_validation=operational_validation,
        distribution_validation=distribution_validation,
        optimization_validation=optimization_validation,
        summary=summary,
        plan=plan,
        alerts=alerts,
    )

    failed = validation.loc[validation["resultado"].eq("FALHA")]
    if not failed.empty:
        details = "; ".join(
            f"{row['teste']} ({row['detalhes']})"
            for _, row in failed.iterrows()
        )
        raise ValueError("Falha na consolidação final: " + details)

    report = render_operational_report(
        ranking=ranking,
        summary=summary,
        future_context=future_context,
        capacities=capacities,
        day_summary=day_summary,
        fair_summary=fair_summary,
        recommendations=recommendations,
        alerts=alerts,
        validation=validation,
    )

    return {
        SUMMARY_KEY: summary,
        PLAN_KEY: plan,
        ALERTS_KEY: alerts,
        VALIDATION_KEY: validation,
        REPORT_KEY: report,
    }


def save_final_consolidation_outputs(
    artifacts: dict[str, FinalArtifact],
    output_directory: str | Path | None = None,
) -> list[Path]:
    """Salva quatro CSVs e o relatório Markdown da Fase 14."""

    if output_directory is None:
        project_root = Path(__file__).resolve().parent.parent
        destination = project_root / "reports" / "final"
    else:
        destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)

    required_keys = {SUMMARY_KEY, PLAN_KEY, ALERTS_KEY, VALIDATION_KEY, REPORT_KEY}
    missing = sorted(required_keys - set(artifacts))
    if missing:
        raise ValueError(
            "Artefatos finais ausentes para salvamento: " + ", ".join(missing)
        )

    generated: list[Path] = []
    for key in [SUMMARY_KEY, PLAN_KEY, ALERTS_KEY, VALIDATION_KEY]:
        dataframe = artifacts[key]
        if not isinstance(dataframe, pd.DataFrame):
            raise TypeError(f"O artefato {key} deveria ser um DataFrame.")
        path = destination / f"{key}.csv"
        dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        generated.append(path)

    report = artifacts[REPORT_KEY]
    if not isinstance(report, str) or not report.strip():
        raise TypeError("O relatório operacional final está vazio.")
    report_path = destination / f"{REPORT_KEY}.md"
    report_path.write_text(report, encoding="utf-8")
    generated.append(report_path)

    return generated

