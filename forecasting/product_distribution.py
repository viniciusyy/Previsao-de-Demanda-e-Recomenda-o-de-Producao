"""
Distribuição das previsões de categoria entre os produtos — Fase 12.

As previsões operacionais de categoria + feira são distribuídas segundo a
participação histórica de cada produto dentro da respectiva feira e categoria.
O método dos maiores restos transforma as parcelas em inteiros sem alterar o
total previsto para a categoria.
"""

from math import floor, isclose, isfinite
from pathlib import Path

import pandas as pd


EXPECTED_FAIRS = 6
EXPECTED_CATEGORIES = 3
EXPECTED_PRODUCTS = 27
EXPECTED_ROWS = EXPECTED_FAIRS * EXPECTED_PRODUCTS
EXPECTED_ALLOWED_COMBINATIONS = 158
EXPECTED_RESTRICTED_COMBINATIONS = 4

FAIR_ORDER = ["QUA", "QUI", "SAB_C", "SAB_E", "DOM_C", "DOM_E"]

RESTRICTED_PRODUCTS_BY_FAIR = {
    "SAB_E": {5, 15},
    "DOM_E": {5, 15},
}

RESTRICTION_REASON = {
    5: "Produto Frango não é comercializado nesta feira.",
    15: "Produto Escarola sem bacon não é comercializado nesta feira.",
}


def _is_allowed(fair: str, product_id: int) -> bool:
    """Informa se o produto pode ser comercializado na feira."""

    return product_id not in RESTRICTED_PRODUCTS_BY_FAIR.get(fair, set())


def prepare_product_history(
    source_data: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Valida o histórico e cria o catálogo único dos produtos."""

    if source_data.empty:
        raise ValueError("A base histórica de produtos está vazia.")

    required_columns = [
        "data_venda",
        "feira",
        "id_produto",
        "produto",
        "categoria",
        "quantidade_vendida",
    ]
    missing_columns = [
        column for column in required_columns if column not in source_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(missing_columns)
        )

    history = source_data[required_columns].copy()
    history["data_venda"] = pd.to_datetime(
        history["data_venda"], errors="raise"
    ).dt.normalize()
    history["id_produto"] = pd.to_numeric(
        history["id_produto"], errors="raise"
    ).astype(int)
    history["quantidade_vendida"] = pd.to_numeric(
        history["quantidade_vendida"], errors="raise"
    )

    if history["quantidade_vendida"].lt(0).any():
        raise ValueError("Foram encontradas quantidades vendidas negativas.")

    history = history.loc[history["data_venda"] <= cutoff_date].copy()

    if history.empty:
        raise ValueError("Não há histórico de produtos até a data de corte.")

    identity_counts = history.groupby("id_produto").agg(
        nomes=("produto", "nunique"),
        categorias=("categoria", "nunique"),
    )
    inconsistent = identity_counts.loc[
        identity_counts["nomes"].ne(1)
        | identity_counts["categorias"].ne(1)
    ]

    if not inconsistent.empty:
        raise ValueError(
            "Produtos com nome ou categoria inconsistentes: "
            + ", ".join(inconsistent.index.astype(str))
        )

    catalog = (
        history[["id_produto", "produto", "categoria"]]
        .drop_duplicates()
        .sort_values("id_produto")
        .reset_index(drop=True)
    )

    if len(catalog) != EXPECTED_PRODUCTS:
        raise ValueError(
            f"Eram esperados {EXPECTED_PRODUCTS} produtos, "
            f"mas foram encontrados {len(catalog)}."
        )

    return history, catalog


def prepare_category_forecasts(
    category_forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Valida as 18 previsões da Fase 11 usadas como totais-alvo."""

    if category_forecasts.empty:
        raise ValueError("As previsões de categoria + feira estão vazias.")

    required_columns = [
        "data_corte",
        "data_producao_prevista",
        "data_venda_prevista",
        "feira",
        "categoria",
        "previsao_operacional",
        "modelo",
        "nome_modelo",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in category_forecasts.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes nas previsões: "
            + ", ".join(missing_columns)
        )

    forecasts = category_forecasts[required_columns].copy()

    for column in [
        "data_corte",
        "data_producao_prevista",
        "data_venda_prevista",
    ]:
        forecasts[column] = pd.to_datetime(
            forecasts[column], errors="raise"
        ).dt.normalize()

    forecasts["previsao_operacional"] = pd.to_numeric(
        forecasts["previsao_operacional"], errors="raise"
    )

    cutoff_dates = forecasts["data_corte"].drop_duplicates()

    if len(cutoff_dates) != 1:
        raise ValueError("As previsões possuem mais de uma data de corte.")

    expected_forecasts = EXPECTED_FAIRS * EXPECTED_CATEGORIES
    duplicated = forecasts.duplicated(subset=["feira", "categoria"])

    if len(forecasts) != expected_forecasts or duplicated.any():
        raise ValueError(
            "Eram esperadas 18 previsões únicas de categoria + feira."
        )

    if forecasts["previsao_operacional"].lt(0).any():
        raise ValueError("Foram encontradas previsões negativas.")

    integer_forecasts = forecasts["previsao_operacional"].map(
        lambda value: float(value).is_integer()
    )

    if not integer_forecasts.all():
        raise ValueError(
            "As previsões operacionais de categoria devem ser inteiras."
        )

    forecasts["previsao_operacional"] = forecasts[
        "previsao_operacional"
    ].astype(int)

    return forecasts, pd.Timestamp(cutoff_dates.iloc[0])


def create_historical_proportions(
    history: pd.DataFrame,
    catalog: pd.DataFrame,
    category_forecasts: pd.DataFrame,
) -> pd.DataFrame:
    """Calcula a participação histórica de cada produto por feira."""

    forecast_context = category_forecasts[
        [
            "data_corte",
            "data_producao_prevista",
            "data_venda_prevista",
            "feira",
            "categoria",
            "previsao_operacional",
            "modelo",
            "nome_modelo",
        ]
    ].rename(
        columns={
            "previsao_operacional": "previsao_categoria",
            "modelo": "modelo_previsao_categoria",
            "nome_modelo": "nome_modelo_previsao_categoria",
        }
    )

    product_grid = forecast_context.merge(
        catalog,
        on="categoria",
        how="inner",
        validate="many_to_many",
    )

    historical_sales = (
        history.groupby(
            ["feira", "categoria", "id_produto"],
            as_index=False,
        )
        .agg(
            vendas_historicas_produto=("quantidade_vendida", "sum"),
            observacoes_historicas_produto=("data_venda", "nunique"),
            primeira_data_historica=("data_venda", "min"),
            ultima_data_historica=("data_venda", "max"),
        )
    )

    distribution = product_grid.merge(
        historical_sales,
        on=["feira", "categoria", "id_produto"],
        how="left",
        validate="one_to_one",
    )

    distribution["vendas_historicas_produto"] = distribution[
        "vendas_historicas_produto"
    ].fillna(0.0)
    distribution["observacoes_historicas_produto"] = distribution[
        "observacoes_historicas_produto"
    ].fillna(0).astype(int)
    distribution["produto_permitido"] = [
        int(_is_allowed(str(fair), int(product_id)))
        for fair, product_id in zip(
            distribution["feira"],
            distribution["id_produto"],
        )
    ]
    distribution["motivo_restricao"] = [
        "" if allowed else RESTRICTION_REASON.get(int(product_id), "Restrição comercial")
        for allowed, product_id in zip(
            distribution["produto_permitido"].astype(bool),
            distribution["id_produto"],
        )
    ]

    proportion_frames = []

    for _, group in distribution.groupby(
        ["feira", "categoria"], sort=False
    ):
        current = group.copy()
        allowed_mask = current["produto_permitido"].eq(1)
        allowed_count = int(allowed_mask.sum())
        allowed_sales = float(
            current.loc[allowed_mask, "vendas_historicas_produto"].sum()
        )

        if allowed_count == 0:
            raise ValueError(
                "Uma combinação feira + categoria ficou sem produtos permitidos."
            )

        current["vendas_historicas_categoria"] = allowed_sales
        current["quantidade_produtos_permitidos_categoria"] = allowed_count
        current["proporcao_historica"] = 0.0

        if allowed_sales > 0:
            current.loc[allowed_mask, "proporcao_historica"] = (
                current.loc[allowed_mask, "vendas_historicas_produto"]
                / allowed_sales
            )
            current["metodo_proporcao"] = "participacao_historica_de_vendas"
        else:
            current.loc[allowed_mask, "proporcao_historica"] = (
                1.0 / allowed_count
            )
            current["metodo_proporcao"] = (
                "distribuicao_uniforme_por_ausencia_de_vendas"
            )

        current["participacao_percentual"] = (
            current["proporcao_historica"] * 100
        )
        proportion_frames.append(current)

    result = pd.concat(proportion_frames, ignore_index=True)
    result["serie_produto"] = (
        result["feira"].astype(str)
        + " | "
        + result["id_produto"].astype(str)
        + " - "
        + result["produto"].astype(str)
    )

    fair_rank = {fair: position for position, fair in enumerate(FAIR_ORDER)}
    result["_ordem_feira"] = result["feira"].map(fair_rank)
    return result.sort_values(
        ["_ordem_feira", "categoria", "id_produto"]
    ).drop(columns="_ordem_feira").reset_index(drop=True)


def allocate_integer_forecasts(
    proportions: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica o método dos maiores restos em cada categoria + feira."""

    allocated_frames = []
    adjustment_rows = []

    for (fair, category), group in proportions.groupby(
        ["feira", "categoria"], sort=False
    ):
        current = group.copy()
        category_forecast = int(current["previsao_categoria"].iloc[0])
        current["alocacao_bruta"] = (
            current["previsao_categoria"]
            * current["proporcao_historica"]
        )
        current["alocacao_piso"] = current["alocacao_bruta"].map(floor)
        current["parte_decimal"] = (
            current["alocacao_bruta"] - current["alocacao_piso"]
        )
        current["ajuste_maior_resto"] = 0
        current["ordem_prioridade_arredondamento"] = pd.Series(
            pd.NA,
            index=current.index,
            dtype="Int64",
        )

        allowed = current.loc[current["produto_permitido"].eq(1)].copy()
        allowed = allowed.sort_values(
            [
                "parte_decimal",
                "vendas_historicas_produto",
                "id_produto",
            ],
            ascending=[False, False, True],
        )
        priority = list(range(1, len(allowed) + 1))
        current.loc[
            allowed.index,
            "ordem_prioridade_arredondamento",
        ] = priority

        floor_sum = int(current["alocacao_piso"].sum())
        residual_units = category_forecast - floor_sum

        if residual_units < 0 or residual_units > len(allowed):
            raise ValueError(
                "Quantidade inválida de unidades residuais para "
                f"{fair} | {category}: {residual_units}."
            )

        receiving_indices = allowed.head(residual_units).index
        current.loc[receiving_indices, "ajuste_maior_resto"] = 1
        current["previsao_produto"] = (
            current["alocacao_piso"] + current["ajuste_maior_resto"]
        ).astype(int)

        restricted_mask = current["produto_permitido"].eq(0)
        current.loc[
            restricted_mask,
            [
                "alocacao_bruta",
                "alocacao_piso",
                "parte_decimal",
                "ajuste_maior_resto",
                "previsao_produto",
            ],
        ] = 0

        receiving_products = current.loc[
            current["ajuste_maior_resto"].eq(1),
            "produto",
        ].astype(str).tolist()

        adjustment_rows.append(
            {
                "feira": fair,
                "categoria": category,
                "previsao_categoria": category_forecast,
                "soma_pisos": floor_sum,
                "unidades_residuais": residual_units,
                "produtos_com_acrescimo": " | ".join(receiving_products),
                "soma_final_produtos": int(
                    current["previsao_produto"].sum()
                ),
                "metodo": "maiores restos",
            }
        )
        allocated_frames.append(current)

    allocations = pd.concat(allocated_frames, ignore_index=True)
    adjustments = pd.DataFrame(adjustment_rows)

    fair_rank = {fair: position for position, fair in enumerate(FAIR_ORDER)}
    allocations["_ordem_feira"] = allocations["feira"].map(fair_rank)
    allocations = allocations.sort_values(
        ["_ordem_feira", "categoria", "id_produto"]
    ).drop(columns="_ordem_feira").reset_index(drop=True)

    adjustments["_ordem_feira"] = adjustments["feira"].map(fair_rank)
    adjustments = adjustments.sort_values(
        ["_ordem_feira", "categoria"]
    ).drop(columns="_ordem_feira").reset_index(drop=True)

    return allocations, adjustments


def create_distribution_summary(
    allocations: pd.DataFrame,
) -> pd.DataFrame:
    """Resume a reconciliação de cada previsão de categoria."""

    summary = (
        allocations.groupby(
            [
                "data_corte",
                "data_producao_prevista",
                "data_venda_prevista",
                "feira",
                "categoria",
            ],
            as_index=False,
        )
        .agg(
            previsao_categoria=("previsao_categoria", "first"),
            previsao_distribuida=("previsao_produto", "sum"),
            produtos_catalogo=("id_produto", "nunique"),
            produtos_permitidos=("produto_permitido", "sum"),
            produtos_com_previsao_positiva=(
                "previsao_produto",
                lambda values: int((values > 0).sum()),
            ),
            unidades_adicionadas_maiores_restos=(
                "ajuste_maior_resto",
                "sum",
            ),
        )
    )
    summary["diferenca_reconciliacao"] = (
        summary["previsao_distribuida"]
        - summary["previsao_categoria"]
    )

    fair_rank = {fair: position for position, fair in enumerate(FAIR_ORDER)}
    summary["_ordem_feira"] = summary["feira"].map(fair_rank)
    return summary.sort_values(
        ["_ordem_feira", "categoria"]
    ).drop(columns="_ordem_feira").reset_index(drop=True)


def create_restriction_table(
    allocations: pd.DataFrame,
) -> pd.DataFrame:
    """Registra as quatro combinações comerciais fixadas em zero."""

    columns = [
        "data_venda_prevista",
        "feira",
        "id_produto",
        "produto",
        "categoria",
        "produto_permitido",
        "motivo_restricao",
        "proporcao_historica",
        "previsao_produto",
    ]
    return allocations.loc[
        allocations["produto_permitido"].eq(0), columns
    ].reset_index(drop=True)


def create_distribution_configuration() -> pd.DataFrame:
    """Documenta as decisões metodológicas da Fase 12."""

    rows = [
        (
            "fonte_total_categoria",
            "previsão operacional da Fase 11",
            "Mantém a soma prevista em categoria + feira.",
        ),
        (
            "proporcao_produto",
            "participação nas vendas históricas da feira e categoria",
            "Preserva o mix observado separadamente em cada feira.",
        ),
        (
            "periodo_proporcao",
            "todo o histórico disponível até a data de corte",
            "Reduz a instabilidade das participações com poucos dados.",
        ),
        (
            "fallback_sem_vendas",
            "proporção uniforme entre produtos permitidos",
            "Evita divisão por zero sem incluir produtos proibidos.",
        ),
        (
            "reconciliacao_inteiros",
            "método dos maiores restos",
            "Garante que a soma dos produtos seja igual à categoria.",
        ),
        (
            "desempate_maiores_restos",
            "vendas históricas decrescentes e id do produto crescente",
            "Mantém o procedimento determinístico e reproduzível.",
        ),
        (
            "restricoes_comerciais",
            "ids 5 e 15 iguais a zero em SAB_E e DOM_E",
            "Registra explicitamente produtos não comercializados.",
        ),
        (
            "escopo",
            "distribuição de previsão; não é recomendação final",
            "O Simplex e demais restrições produtivas serão posteriores.",
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=["parametro", "valor", "justificativa"],
    )


def create_distribution_decision(
    allocations: pd.DataFrame,
    source_record_count: int,
) -> pd.DataFrame:
    """Registra o snapshot e a decisão operacional da distribuição."""

    return pd.DataFrame(
        [
            {
                "registros_origem_mysql": int(source_record_count),
                "data_corte": allocations["data_corte"].max(),
                "modelo_previsao_categoria": allocations[
                    "modelo_previsao_categoria"
                ].iloc[0],
                "metodo_distribuicao": (
                    "proporções históricas por feira e categoria"
                ),
                "metodo_reconciliacao": "maiores restos",
                "quantidade_feiras": allocations["feira"].nunique(),
                "quantidade_categorias": allocations[
                    "categoria"
                ].nunique(),
                "quantidade_produtos": allocations["id_produto"].nunique(),
                "linhas_produto_feira": len(allocations),
                "combinacoes_permitidas": int(
                    allocations["produto_permitido"].sum()
                ),
                "combinacoes_restritas": int(
                    allocations["produto_permitido"].eq(0).sum()
                ),
                "previsao_total_categorias": int(
                    allocations.groupby(["feira", "categoria"])[
                        "previsao_categoria"
                    ].first().sum()
                ),
                "previsao_total_produtos": int(
                    allocations["previsao_produto"].sum()
                ),
                "recomendacao_final_producao_gerada": "não",
                "status": (
                    "Previsões distribuídas entre produtos e reconciliadas "
                    "com os totais das categorias."
                ),
            }
        ]
    )


def validate_product_distribution(
    allocations: pd.DataFrame,
    adjustments: pd.DataFrame,
    category_forecasts: pd.DataFrame,
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Audita cobertura, proporções, restrições e reconciliação."""

    checks: list[dict[str, str]] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "teste": name,
                "resultado": "OK" if passed else "FALHA",
                "detalhes": details,
            }
        )

    add_check(
        "Catálogo completo nas seis feiras",
        len(allocations) == EXPECTED_ROWS
        and allocations["id_produto"].nunique() == EXPECTED_PRODUCTS
        and allocations["feira"].nunique() == EXPECTED_FAIRS,
        (
            f"Linhas: {len(allocations)}; produtos: "
            f"{allocations['id_produto'].nunique()}; feiras: "
            f"{allocations['feira'].nunique()}"
        ),
    )

    allowed_count = int(allocations["produto_permitido"].sum())
    restricted_count = int(allocations["produto_permitido"].eq(0).sum())
    add_check(
        "Combinações permitidas e restritas",
        allowed_count == EXPECTED_ALLOWED_COMBINATIONS
        and restricted_count == EXPECTED_RESTRICTED_COMBINATIONS,
        f"Permitidas: {allowed_count}; restritas: {restricted_count}",
    )

    duplicated = allocations.duplicated(
        subset=["data_venda_prevista", "feira", "id_produto"]
    ).sum()
    add_check(
        "Chave futura produto + feira sem duplicidades",
        duplicated == 0,
        f"Duplicidades encontradas: {int(duplicated)}",
    )

    expected_groups = set(
        zip(category_forecasts["feira"], category_forecasts["categoria"])
    )
    found_groups = set(zip(allocations["feira"], allocations["categoria"]))
    add_check(
        "Cobertura das 18 previsões de categoria",
        found_groups == expected_groups and len(found_groups) == 18,
        f"Grupos esperados: {len(expected_groups)}; encontrados: {len(found_groups)}",
    )

    proportion_sums = allocations.groupby(
        ["feira", "categoria"]
    )["proporcao_historica"].sum()
    proportions_sum_one = all(
        isclose(float(value), 1.0, abs_tol=1e-10)
        for value in proportion_sums
    )
    add_check(
        "Proporções somam um por categoria",
        proportions_sum_one,
        f"Intervalo das somas: {proportion_sums.min():.10f} a {proportion_sums.max():.10f}",
    )

    valid_proportions = allocations["proporcao_historica"].between(0, 1).all()
    finite_proportions = all(
        isfinite(float(value))
        for value in allocations["proporcao_historica"].tolist()
    )
    add_check(
        "Proporções válidas",
        bool(valid_proportions and finite_proportions),
        "As proporções devem ser finitas e permanecer entre zero e um.",
    )

    restricted = allocations.loc[allocations["produto_permitido"].eq(0)]
    restricted_zero = (
        restricted["proporcao_historica"].eq(0).all()
        and restricted["previsao_produto"].eq(0).all()
        and set(restricted["id_produto"]) == {5, 15}
        and set(restricted["feira"]) == {"SAB_E", "DOM_E"}
    )
    add_check(
        "Restrições comerciais fixadas em zero",
        bool(restricted_zero),
        f"Combinações restritas auditadas: {len(restricted)}",
    )

    restricted_history = history.loc[
        history["feira"].isin(["SAB_E", "DOM_E"])
        & history["id_produto"].isin([5, 15])
    ]
    restricted_positive = int(
        restricted_history["quantidade_vendida"].gt(0).sum()
    )
    add_check(
        "Histórico sem venda positiva proibida",
        restricted_positive == 0,
        f"Registros positivos proibidos: {restricted_positive}",
    )

    expected_raw = (
        allocations["previsao_categoria"]
        * allocations["proporcao_historica"]
    )
    raw_allocation_ok = all(
        isclose(float(found), float(expected), abs_tol=1e-10)
        for found, expected in zip(
            allocations["alocacao_bruta"], expected_raw
        )
    )
    add_check(
        "Alocação bruta calculada corretamente",
        raw_allocation_ok,
        "Cada parcela deve ser a previsão da categoria multiplicada pela proporção.",
    )

    final_values = allocations["previsao_produto"]
    integer_nonnegative = final_values.ge(0).all() and all(
        float(value).is_integer() for value in final_values
    )
    add_check(
        "Previsões por produto inteiras e não negativas",
        bool(integer_nonnegative),
        f"Menor previsão encontrada: {int(final_values.min())}",
    )

    distributed_totals = allocations.groupby(
        ["feira", "categoria"]
    )["previsao_produto"].sum()
    category_totals = allocations.groupby(
        ["feira", "categoria"]
    )["previsao_categoria"].first()
    exact_reconciliation = distributed_totals.equals(
        category_totals.astype(int)
    )
    add_check(
        "Soma dos produtos igual à previsão da categoria",
        exact_reconciliation,
        f"Maior diferença: {int((distributed_totals - category_totals).abs().max())}",
    )

    adjustment_values_ok = allocations["ajuste_maior_resto"].isin([0, 1]).all()
    residuals_match = adjustments["unidades_residuais"].equals(
        adjustments["soma_final_produtos"] - adjustments["soma_pisos"]
    )
    add_check(
        "Reconciliação pelo método dos maiores restos",
        bool(adjustment_values_ok and residuals_match),
        "Cada unidade residual deve ser atribuída uma única vez.",
    )

    product_counts = allocations.groupby("feira")["id_produto"].nunique()
    add_check(
        "Vinte e sete produtos registrados por feira",
        len(product_counts) == EXPECTED_FAIRS
        and product_counts.eq(EXPECTED_PRODUCTS).all(),
        f"Produtos por feira: {product_counts.to_dict()}",
    )

    future_dates_match = all(
        group["data_venda_prevista"].nunique() == 1
        and group["data_producao_prevista"].nunique() == 1
        for _, group in allocations.groupby(["feira", "categoria"])
    )
    add_check(
        "Datas da Fase 11 preservadas",
        future_dates_match,
        "Os produtos devem herdar as datas da respectiva categoria e feira.",
    )

    history_before_cutoff = (
        history["data_venda"] <= allocations["data_corte"].max()
    ).all()
    add_check(
        "Proporções usam somente histórico disponível",
        bool(history_before_cutoff),
        "Nenhuma venda posterior à data de corte pode participar das proporções.",
    )

    valid_methods = {
        "participacao_historica_de_vendas",
        "distribuicao_uniforme_por_ausencia_de_vendas",
    }
    methods_ok = set(allocations["metodo_proporcao"]).issubset(valid_methods)
    add_check(
        "Método de proporção documentado",
        methods_ok,
        "Métodos encontrados: "
        + ", ".join(sorted(set(allocations["metodo_proporcao"]))),
    )

    total_category = int(category_totals.sum())
    total_products = int(final_values.sum())
    add_check(
        "Total geral preservado",
        total_category == total_products,
        f"Categorias: {total_category}; produtos: {total_products}",
    )

    return pd.DataFrame(checks)


def run_product_distribution(
    source_data: pd.DataFrame,
    category_forecasts: pd.DataFrame,
    source_record_count: int,
) -> dict[str, pd.DataFrame]:
    """Executa a distribuição das previsões entre os produtos."""

    forecasts, cutoff_date = prepare_category_forecasts(category_forecasts)
    history, catalog = prepare_product_history(source_data, cutoff_date)
    proportions = create_historical_proportions(
        history,
        catalog,
        forecasts,
    )
    allocations, adjustments = allocate_integer_forecasts(proportions)
    summary = create_distribution_summary(allocations)
    restrictions = create_restriction_table(allocations)
    validation = validate_product_distribution(
        allocations,
        adjustments,
        forecasts,
        history,
    )

    if validation["resultado"].eq("FALHA").any():
        failed_rows = validation.loc[
            validation["resultado"] == "FALHA",
            ["teste", "detalhes"],
        ]
        failed = [
            f"{row['teste']} ({row['detalhes']})"
            for _, row in failed_rows.iterrows()
        ]
        raise ValueError(
            "Falha na distribuição por produto: " + "; ".join(failed)
        )

    proportion_columns = [
        "data_corte",
        "feira",
        "categoria",
        "id_produto",
        "produto",
        "produto_permitido",
        "motivo_restricao",
        "vendas_historicas_produto",
        "vendas_historicas_categoria",
        "observacoes_historicas_produto",
        "primeira_data_historica",
        "ultima_data_historica",
        "proporcao_historica",
        "participacao_percentual",
        "metodo_proporcao",
    ]

    return {
        "previsoes_operacionais_produto_feira": allocations,
        "proporcoes_historicas_produto_feira": allocations[
            proportion_columns
        ].copy(),
        "resumo_distribuicao_produtos": summary,
        "ajustes_arredondamento_produtos": adjustments,
        "restricoes_comerciais_produtos": restrictions,
        "validacao_distribuicao_produtos": validation,
        "configuracao_distribuicao_produtos": (
            create_distribution_configuration()
        ),
        "decisao_distribuicao_produtos": create_distribution_decision(
            allocations,
            source_record_count,
        ),
    }


def save_product_distribution_outputs(
    analyses: dict[str, pd.DataFrame],
) -> list[Path]:
    """Salva as previsões por produto e os relatórios da Fase 12."""

    project_root = Path(__file__).resolve().parent.parent
    forecasts_directory = project_root / "reports" / "forecasts"
    tables_directory = project_root / "reports" / "tables"

    forecasts_directory.mkdir(parents=True, exist_ok=True)
    tables_directory.mkdir(parents=True, exist_ok=True)

    forecast_files = {
        "previsoes_operacionais_produto_feira",
        "proporcoes_historicas_produto_feira",
    }
    generated_files = []

    for name, dataframe in analyses.items():
        directory = (
            forecasts_directory if name in forecast_files else tables_directory
        )
        path = directory / f"{name}.csv"
        dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        generated_files.append(path)

    return generated_files
