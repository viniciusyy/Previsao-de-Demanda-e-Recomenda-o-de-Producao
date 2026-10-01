"""
Recomendação de produção por Programação Linear.

As previsões por produto + feira produzidas são transformadas em
recomendações sujeitas a um limite operacional por dia de produção. Na
configuração atual, esse limite corresponde ao maior volume total produzido
em uma mesma data histórica para o respectivo grupo operacional.

O problema linear maximiza o atendimento da demanda prevista, sem recomendar
quantidades superiores às previsões e preservando proporcionalmente o mix de
produtos de cada dia. O modelo contínuo é resolvido pelo dual Simplex do HiGHS
e, em seguida, convertido em unidades inteiras pelo método dos maiores restos.

A obtenção das capacidades foi isolada da formulação para permitir que, no
futuro, os limites históricos sejam substituídos por capacidades informadas,
estoques de ingredientes ou outros recursos produtivos.
"""

from __future__ import annotations

from math import floor, isclose, isfinite
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd
from scipy.optimize import linprog


EXPECTED_FAIRS = 6
EXPECTED_PRODUCTS = 27
EXPECTED_ROWS = EXPECTED_FAIRS * EXPECTED_PRODUCTS

CAPACITY_SOURCE_HISTORICAL_MAX = "historical_max"
CAPACITY_SOURCE_MANUAL = "manual"
DEFAULT_CAPACITY_SOURCE = CAPACITY_SOURCE_HISTORICAL_MAX
SOLVER_METHOD = "highs-ds"
NUMERICAL_TOLERANCE = 1e-7

FAIR_ORDER = ["QUA", "QUI", "SAB_C", "SAB_E", "DOM_C", "DOM_E"]

PRODUCTION_GROUP_BY_FAIR = {
    "QUA": "TERCA_QUA",
    "QUI": "QUARTA_QUI",
    "SAB_C": "SEXTA_SABADOS",
    "SAB_E": "SEXTA_SABADOS",
    "DOM_C": "SABADO_DOMINGOS",
    "DOM_E": "SABADO_DOMINGOS",
}

PRODUCTION_GROUP_CONFIGURATION = {
    "TERCA_QUA": {
        "dia_producao": "terça-feira",
        "feiras": "QUA",
        "ordem": 1,
    },
    "QUARTA_QUI": {
        "dia_producao": "quarta-feira",
        "feiras": "QUI",
        "ordem": 2,
    },
    "SEXTA_SABADOS": {
        "dia_producao": "sexta-feira",
        "feiras": "SAB_C + SAB_E",
        "ordem": 3,
    },
    "SABADO_DOMINGOS": {
        "dia_producao": "sábado",
        "feiras": "DOM_C + DOM_E",
        "ordem": 4,
    },
}


def _production_group_sort(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Ordena um DataFrame pela sequência operacional dos dias de produção."""

    result = dataframe.copy()
    order = {
        group: configuration["ordem"]
        for group, configuration in PRODUCTION_GROUP_CONFIGURATION.items()
    }
    result["_ordem_grupo"] = result["grupo_producao"].map(order)
    sort_columns = ["_ordem_grupo"]

    for optional_column in ["feira", "categoria", "id_produto"]:
        if optional_column in result.columns:
            sort_columns.append(optional_column)

    return (
        result.sort_values(sort_columns)
        .drop(columns="_ordem_grupo")
        .reset_index(drop=True)
    )


def prepare_product_forecasts(
    product_forecasts: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Valida as previsões por produto que serão usadas pela otimização."""

    if product_forecasts.empty:
        raise ValueError("As previsões por produto + feira estão vazias.")

    required_columns = [
        "data_corte",
        "data_producao_prevista",
        "data_venda_prevista",
        "feira",
        "categoria",
        "id_produto",
        "produto",
        "produto_permitido",
        "motivo_restricao",
        "previsao_produto",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in product_forecasts.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes nas previsões por produto: "
            + ", ".join(missing_columns)
        )

    forecasts = product_forecasts.copy()

    for column in [
        "data_corte",
        "data_producao_prevista",
        "data_venda_prevista",
    ]:
        forecasts[column] = pd.to_datetime(
            forecasts[column], errors="raise"
        ).dt.normalize()

    forecasts["id_produto"] = pd.to_numeric(
        forecasts["id_produto"], errors="raise"
    ).astype(int)
    forecasts["produto_permitido"] = pd.to_numeric(
        forecasts["produto_permitido"], errors="raise"
    ).astype(int)
    forecasts["previsao_produto"] = pd.to_numeric(
        forecasts["previsao_produto"], errors="raise"
    )

    if not forecasts["produto_permitido"].isin([0, 1]).all():
        raise ValueError(
            "A coluna produto_permitido deve conter apenas 0 ou 1."
        )

    if forecasts["previsao_produto"].lt(0).any():
        raise ValueError("Foram encontradas previsões negativas por produto.")

    integer_forecasts = forecasts["previsao_produto"].map(
        lambda value: float(value).is_integer()
    )
    if not integer_forecasts.all():
        raise ValueError("As previsões por produto devem ser inteiras.")

    forecasts["previsao_produto"] = forecasts[
        "previsao_produto"
    ].astype(int)

    cutoff_dates = forecasts["data_corte"].drop_duplicates()
    if len(cutoff_dates) != 1:
        raise ValueError("As previsões possuem mais de uma data de corte.")

    cutoff_date = pd.Timestamp(cutoff_dates.iloc[0])

    unknown_fairs = sorted(
        set(forecasts["feira"]) - set(PRODUCTION_GROUP_BY_FAIR)
    )
    if unknown_fairs:
        raise ValueError(
            "Feiras sem grupo de produção configurado: "
            + ", ".join(unknown_fairs)
        )

    duplicated = forecasts.duplicated(
        subset=["data_venda_prevista", "feira", "id_produto"]
    )
    if duplicated.any():
        raise ValueError(
            "Existem duplicidades na chave futura data + feira + produto."
        )

    if len(forecasts) != EXPECTED_ROWS:
        raise ValueError(
            f"Eram esperadas {EXPECTED_ROWS} linhas produto + feira, "
            f"mas foram encontradas {len(forecasts)}."
        )

    if forecasts["feira"].nunique() != EXPECTED_FAIRS:
        raise ValueError(f"Eram esperadas {EXPECTED_FAIRS} feiras.")

    if forecasts["id_produto"].nunique() != EXPECTED_PRODUCTS:
        raise ValueError(f"Eram esperados {EXPECTED_PRODUCTS} produtos.")

    restricted_positive = forecasts.loc[
        forecasts["produto_permitido"].eq(0),
        "previsao_produto",
    ].gt(0)

    if restricted_positive.any():
        raise ValueError(
            "Uma combinação comercialmente restrita possui previsão positiva."
        )

    forecasts["grupo_producao"] = forecasts["feira"].map(
        PRODUCTION_GROUP_BY_FAIR
    )
    forecasts["dia_producao"] = forecasts["grupo_producao"].map(
        {
            group: configuration["dia_producao"]
            for group, configuration
            in PRODUCTION_GROUP_CONFIGURATION.items()
        }
    )

    date_counts = forecasts.groupby("grupo_producao")[
        "data_producao_prevista"
    ].nunique()

    if not date_counts.eq(1).all():
        raise ValueError(
            "As feiras do mesmo grupo devem compartilhar a data de produção."
        )

    return _production_group_sort(forecasts), cutoff_date


def prepare_capacity_history(
    source_data: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Agrega a produção histórica por data e grupo operacional."""

    if source_data.empty:
        raise ValueError(
            "A base histórica utilizada na capacidade está vazia."
        )

    required_columns = [
        "id_operacao",
        "data_producao",
        "data_venda",
        "feira",
        "id_produto",
        "quantidade_produzida",
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in source_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes no histórico de capacidade: "
            + ", ".join(missing_columns)
        )

    history = source_data[required_columns].copy()
    history["data_producao"] = pd.to_datetime(
        history["data_producao"],
        errors="raise",
    ).dt.normalize()
    history["data_venda"] = pd.to_datetime(
        history["data_venda"],
        errors="raise",
    ).dt.normalize()
    history["id_produto"] = pd.to_numeric(
        history["id_produto"],
        errors="raise",
    ).astype(int)
    history["quantidade_produzida"] = pd.to_numeric(
        history["quantidade_produzida"],
        errors="raise",
    )

    finite_values = history["quantidade_produzida"].map(
        lambda value: isfinite(float(value))
    )

    if (
        not finite_values.all()
        or history["quantidade_produzida"].lt(0).any()
    ):
        raise ValueError(
            "A produção histórica possui valores negativos ou não finitos."
        )

    duplicated = history.duplicated(
        subset=["id_operacao", "id_produto"]
    )
    if duplicated.any():
        raise ValueError(
            "Existem registros duplicados de operação + produto no histórico."
        )

    history = history.loc[
        history["data_venda"] <= cutoff_date
    ].copy()

    if history.empty:
        raise ValueError("Não há produção histórica até a data de corte.")

    unknown_fairs = sorted(
        set(history["feira"]) - set(PRODUCTION_GROUP_BY_FAIR)
    )
    if unknown_fairs:
        raise ValueError(
            "Feiras históricas sem grupo de produção configurado: "
            + ", ".join(unknown_fairs)
        )

    history["grupo_producao"] = history["feira"].map(
        PRODUCTION_GROUP_BY_FAIR
    )

    daily = (
        history.groupby(
            ["grupo_producao", "data_producao"],
            as_index=False,
        )
        .agg(
            producao_total_dia=("quantidade_produzida", "sum"),
            quantidade_operacoes=("id_operacao", "nunique"),
            quantidade_produtos=("id_produto", "nunique"),
            feiras_presentes=(
                "feira",
                lambda values: " + ".join(
                    fair
                    for fair in FAIR_ORDER
                    if fair in set(values.astype(str))
                ),
            ),
            maior_data_venda=("data_venda", "max"),
        )
    )

    daily["producao_total_dia"] = daily[
        "producao_total_dia"
    ].round(10)
    daily["dia_producao"] = daily["grupo_producao"].map(
        {
            group: configuration["dia_producao"]
            for group, configuration
            in PRODUCTION_GROUP_CONFIGURATION.items()
        }
    )
    daily["fonte_capacidade"] = CAPACITY_SOURCE_HISTORICAL_MAX

    return _production_group_sort(daily)


def estimate_historical_capacities(
    capacity_history: pd.DataFrame,
) -> pd.DataFrame:
    """Define a capacidade como o máximo histórico produzido em cada grupo."""

    rows: list[dict[str, object]] = []

    for group, configuration in PRODUCTION_GROUP_CONFIGURATION.items():
        current = capacity_history.loc[
            capacity_history["grupo_producao"] == group
        ].copy()

        if current.empty:
            raise ValueError(
                f"Não há histórico para estimar a capacidade de {group}."
            )

        maximum_index = current["producao_total_dia"].idxmax()
        maximum_row = current.loc[maximum_index]
        maximum_value = float(maximum_row["producao_total_dia"])

        if not maximum_value.is_integer():
            raise ValueError(
                "A capacidade histórica resultou em quantidade não inteira "
                f"para {group}: {maximum_value}."
            )

        rows.append(
            {
                "grupo_producao": group,
                "dia_producao": configuration["dia_producao"],
                "feiras_atendidas": configuration["feiras"],
                "capacidade_estimada": int(maximum_value),
                "quantidade_datas_historicas": len(current),
                "producao_minima_historica": float(
                    current["producao_total_dia"].min()
                ),
                "producao_media_historica": float(
                    current["producao_total_dia"].mean()
                ),
                "producao_mediana_historica": float(
                    current["producao_total_dia"].median()
                ),
                "producao_maxima_historica": maximum_value,
                "data_producao_maxima": maximum_row["data_producao"],
                "fonte_capacidade": CAPACITY_SOURCE_HISTORICAL_MAX,
                "interpretacao": (
                    "limite operacional observado; não representa "
                    "capacidade física definitiva"
                ),
            }
        )

    return _production_group_sort(pd.DataFrame(rows))


def create_manual_capacities(
    manual_capacities: Mapping[str, int | float],
) -> pd.DataFrame:
    """Prepara limites manuais sem alterar a formulação da otimização."""

    expected_groups = set(PRODUCTION_GROUP_CONFIGURATION)
    informed_groups = set(manual_capacities)

    if informed_groups != expected_groups:
        missing = sorted(expected_groups - informed_groups)
        extra = sorted(informed_groups - expected_groups)
        raise ValueError(
            "Capacidades manuais incompletas. Ausentes: "
            f"{missing}; desconhecidas: {extra}."
        )

    rows = []

    for group, configuration in PRODUCTION_GROUP_CONFIGURATION.items():
        value = float(manual_capacities[group])

        if (
            not isfinite(value)
            or value <= 0
            or not value.is_integer()
        ):
            raise ValueError(
                f"A capacidade manual de {group} deve ser inteira e positiva."
            )

        rows.append(
            {
                "grupo_producao": group,
                "dia_producao": configuration["dia_producao"],
                "feiras_atendidas": configuration["feiras"],
                "capacidade_estimada": int(value),
                "quantidade_datas_historicas": pd.NA,
                "producao_minima_historica": np.nan,
                "producao_media_historica": np.nan,
                "producao_mediana_historica": np.nan,
                "producao_maxima_historica": np.nan,
                "data_producao_maxima": pd.NaT,
                "fonte_capacidade": CAPACITY_SOURCE_MANUAL,
                "interpretacao": (
                    "limite operacional informado manualmente"
                ),
            }
        )

    return _production_group_sort(pd.DataFrame(rows))


def resolve_capacities(
    source_data: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    capacity_source: str = DEFAULT_CAPACITY_SOURCE,
    manual_capacities: Mapping[str, int | float] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seleciona a fonte de capacidade sem acoplar o modelo ao seu cálculo."""

    capacity_history = prepare_capacity_history(
        source_data,
        cutoff_date,
    )

    if capacity_source == CAPACITY_SOURCE_HISTORICAL_MAX:
        capacities = estimate_historical_capacities(
            capacity_history
        )
    elif capacity_source == CAPACITY_SOURCE_MANUAL:
        if manual_capacities is None:
            raise ValueError(
                "Capacidades manuais não foram informadas."
            )

        capacities = create_manual_capacities(
            manual_capacities
        )
    else:
        raise ValueError(
            "Fonte de capacidade desconhecida: "
            + str(capacity_source)
        )

    return capacities, capacity_history


def solve_linear_recommendation(
    forecasts: pd.DataFrame,
    capacities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Resolve o modelo contínuo com o dual Simplex do HiGHS."""

    optimization_data = forecasts.reset_index(drop=True).copy()
    groups = list(PRODUCTION_GROUP_CONFIGURATION)
    group_position = {
        group: position
        for position, group in enumerate(groups)
    }

    row_count = len(optimization_data)
    variable_count = row_count + len(groups)

    # scipy.optimize.linprog minimiza. Coeficientes negativos equivalem a
    # maximizar o total recomendado.
    objective = np.zeros(variable_count, dtype=float)
    objective[:row_count] = -1.0

    capacity_by_group = capacities.set_index("grupo_producao")[
        "capacidade_estimada"
    ].astype(float)

    inequality_matrix = np.zeros(
        (len(groups), variable_count),
        dtype=float,
    )
    inequality_limits = np.zeros(len(groups), dtype=float)

    for group, group_index in group_position.items():
        row_indices = optimization_data.index[
            optimization_data["grupo_producao"] == group
        ].tolist()
        inequality_matrix[group_index, row_indices] = 1.0
        inequality_limits[group_index] = capacity_by_group.loc[group]

    # x_i = previsão_i * lambda_g preserva o mix previsto no grupo.
    equality_matrix = np.zeros(
        (row_count, variable_count),
        dtype=float,
    )
    equality_limits = np.zeros(row_count, dtype=float)

    for row_index, row in optimization_data.iterrows():
        group_index = group_position[
            str(row["grupo_producao"])
        ]
        equality_matrix[row_index, row_index] = 1.0
        equality_matrix[
            row_index,
            row_count + group_index,
        ] = -float(row["previsao_produto"])

    bounds = [
        (0.0, float(forecast))
        for forecast in optimization_data["previsao_produto"]
    ] + [
        (0.0, 1.0)
        for _ in groups
    ]

    result = linprog(
        c=objective,
        A_ub=inequality_matrix,
        b_ub=inequality_limits,
        A_eq=equality_matrix,
        b_eq=equality_limits,
        bounds=bounds,
        method=SOLVER_METHOD,
    )

    if not result.success:
        raise RuntimeError(
            "O Simplex não encontrou solução ótima. "
            f"Status {result.status}: {result.message}"
        )

    optimization_data["recomendacao_continua"] = result.x[
        :row_count
    ]

    factor_by_group = {
        group: float(
            result.x[row_count + group_position[group]]
        )
        for group in groups
    }

    optimization_data[
        "fator_atendimento_continuo"
    ] = optimization_data["grupo_producao"].map(
        factor_by_group
    )
    optimization_data[
        "metodo_otimizacao"
    ] = "programacao_linear"
    optimization_data["algoritmo_solver"] = SOLVER_METHOD

    diagnostics = pd.DataFrame(
        [
            {
                "algoritmo": SOLVER_METHOD,
                "sucesso": int(result.success),
                "status_codigo": int(result.status),
                "mensagem": str(result.message),
                "valor_funcao_objetivo_minimizacao": float(
                    result.fun
                ),
                "total_continuo_maximizado": float(-result.fun),
                "quantidade_variaveis_produto_feira": row_count,
                "quantidade_variaveis_fator_atendimento": len(
                    groups
                ),
                "quantidade_restricoes_capacidade": len(groups),
                "quantidade_restricoes_preservacao_mix": row_count,
                "iteracoes_solver": int(
                    getattr(result, "nit", 0)
                ),
            }
        ]
    )

    return optimization_data, diagnostics


def round_recommendations(
    continuous_recommendations: pd.DataFrame,
    capacities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Converte a solução contínua em unidades inteiras reconciliadas."""

    capacity_by_group = capacities.set_index("grupo_producao")[
        "capacidade_estimada"
    ].astype(int)

    rounded_frames: list[pd.DataFrame] = []
    adjustment_rows: list[dict[str, object]] = []

    for group, current_group in continuous_recommendations.groupby(
        "grupo_producao",
        sort=False,
    ):
        current = current_group.copy()
        forecast_total = int(
            current["previsao_produto"].sum()
        )
        capacity = int(capacity_by_group.loc[group])
        target_total = min(forecast_total, capacity)

        current["recomendacao_piso"] = current[
            "recomendacao_continua"
        ].map(
            lambda value: floor(
                float(value) + NUMERICAL_TOLERANCE
            )
        )
        current["parte_decimal_recomendacao"] = (
            current["recomendacao_continua"]
            - current["recomendacao_piso"]
        ).clip(lower=0.0)
        current["ajuste_maior_resto_recomendacao"] = 0

        floor_total = int(
            current["recomendacao_piso"].sum()
        )
        residual_units = target_total - floor_total

        eligible = current.loc[
            current["recomendacao_piso"]
            < current["previsao_produto"]
        ].sort_values(
            [
                "parte_decimal_recomendacao",
                "previsao_produto",
                "feira",
                "id_produto",
            ],
            ascending=[False, False, True, True],
        )

        if (
            residual_units < 0
            or residual_units > len(eligible)
        ):
            raise ValueError(
                "Não foi possível reconciliar o arredondamento do grupo "
                f"{group}. Resíduo: {residual_units}; "
                f"elegíveis: {len(eligible)}."
            )

        receiving_indices = eligible.head(
            residual_units
        ).index

        current.loc[
            receiving_indices,
            "ajuste_maior_resto_recomendacao",
        ] = 1

        current["recomendacao_producao"] = (
            current["recomendacao_piso"]
            + current["ajuste_maior_resto_recomendacao"]
        ).astype(int)

        current["reducao_em_relacao_previsao"] = (
            current["previsao_produto"]
            - current["recomendacao_producao"]
        ).astype(int)

        current["percentual_atendimento_produto"] = np.where(
            current["previsao_produto"] > 0,
            (
                current["recomendacao_producao"]
                / current["previsao_produto"]
                * 100
            ),
            0.0,
        )
        current["capacidade_estimada_grupo"] = capacity
        current["fonte_capacidade"] = capacities.loc[
            capacities["grupo_producao"] == group,
            "fonte_capacidade",
        ].iloc[0]

        adjustment_rows.append(
            {
                "grupo_producao": group,
                "dia_producao": current[
                    "dia_producao"
                ].iloc[0],
                "data_producao_prevista": current[
                    "data_producao_prevista"
                ].iloc[0],
                "previsao_total": forecast_total,
                "capacidade_estimada": capacity,
                "alvo_inteiro_recomendacao": target_total,
                "soma_pisos": floor_total,
                "unidades_residuais": residual_units,
                "soma_final_recomendacao": int(
                    current["recomendacao_producao"].sum()
                ),
                "metodo": (
                    "maiores restos após solução contínua"
                ),
            }
        )
        rounded_frames.append(current)

    recommendations = _production_group_sort(
        pd.concat(
            rounded_frames,
            ignore_index=True,
        )
    )
    adjustments = _production_group_sort(
        pd.DataFrame(adjustment_rows)
    )

    return recommendations, adjustments


def create_recommendation_summaries(
    recommendations: pd.DataFrame,
    capacities: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cria resumos por dia operacional e por feira."""

    capacity_columns = [
        "grupo_producao",
        "feiras_atendidas",
        "capacidade_estimada",
        "quantidade_datas_historicas",
        "producao_maxima_historica",
        "data_producao_maxima",
        "fonte_capacidade",
    ]

    by_group = (
        recommendations.groupby(
            [
                "grupo_producao",
                "dia_producao",
                "data_producao_prevista",
            ],
            as_index=False,
        )
        .agg(
            previsao_total=("previsao_produto", "sum"),
            recomendacao_continua_total=(
                "recomendacao_continua",
                "sum",
            ),
            recomendacao_total=(
                "recomendacao_producao",
                "sum",
            ),
            reducao_total=(
                "reducao_em_relacao_previsao",
                "sum",
            ),
            fator_atendimento_continuo=(
                "fator_atendimento_continuo",
                "first",
            ),
            quantidade_produtos=("id_produto", "count"),
        )
        .merge(
            capacities[capacity_columns],
            on="grupo_producao",
            how="left",
            validate="one_to_one",
        )
    )

    by_group["restricao_capacidade_ativa"] = np.where(
        by_group["previsao_total"]
        > by_group["capacidade_estimada"],
        "sim",
        "não",
    )
    by_group["capacidade_ociosa"] = (
        by_group["capacidade_estimada"]
        - by_group["recomendacao_total"]
    )
    by_group["percentual_atendimento"] = np.where(
        by_group["previsao_total"] > 0,
        (
            by_group["recomendacao_total"]
            / by_group["previsao_total"]
            * 100
        ),
        100.0,
    )
    by_group[
        "percentual_utilizacao_capacidade"
    ] = np.where(
        by_group["capacidade_estimada"] > 0,
        (
            by_group["recomendacao_total"]
            / by_group["capacidade_estimada"]
            * 100
        ),
        0.0,
    )
    by_group = _production_group_sort(by_group)

    by_fair = (
        recommendations.groupby(
            [
                "data_producao_prevista",
                "data_venda_prevista",
                "grupo_producao",
                "dia_producao",
                "feira",
            ],
            as_index=False,
        )
        .agg(
            previsao_total=("previsao_produto", "sum"),
            recomendacao_total=(
                "recomendacao_producao",
                "sum",
            ),
            reducao_total=(
                "reducao_em_relacao_previsao",
                "sum",
            ),
            produtos_com_previsao=(
                "previsao_produto",
                lambda values: int((values > 0).sum()),
            ),
            produtos_com_recomendacao=(
                "recomendacao_producao",
                lambda values: int((values > 0).sum()),
            ),
        )
    )

    by_fair["percentual_atendimento"] = np.where(
        by_fair["previsao_total"] > 0,
        (
            by_fair["recomendacao_total"]
            / by_fair["previsao_total"]
            * 100
        ),
        100.0,
    )

    fair_rank = {
        fair: position
        for position, fair in enumerate(FAIR_ORDER)
    }
    by_fair["_ordem_feira"] = by_fair["feira"].map(
        fair_rank
    )
    by_fair = (
        by_fair.sort_values("_ordem_feira")
        .drop(columns="_ordem_feira")
        .reset_index(drop=True)
    )

    return by_group, by_fair


def create_optimization_configuration(
    capacity_source: str,
) -> pd.DataFrame:
    """Documenta as decisões metodológicas e seu caráter provisório."""

    rows = [
        (
            "variavel_decisao",
            "quantidade contínua recomendada por produto + feira",
            "A conversão para unidades inteiras ocorre após o Simplex.",
        ),
        (
            "funcao_objetivo",
            "maximizar o total atendido da demanda prevista",
            "Reduz a demanda prevista não atendida sob capacidade limitada.",
        ),
        (
            "limite_superior_produto",
            "recomendação não pode superar a previsão",
            "Evita criar produção adicional sem sustentação na previsão.",
        ),
        (
            "preservacao_mix",
            "mesmo fator de atendimento no grupo de produção",
            "Evita priorização arbitrária entre produtos e feiras.",
        ),
        (
            "fonte_capacidade_ativa",
            capacity_source,
            "A fonte foi isolada e poderá ser substituída futuramente.",
        ),
        (
            "capacidade_historica",
            (
                "maior total produzido em uma mesma data "
                "por grupo operacional"
            ),
            (
                "Representa limite observado, "
                "não capacidade física definitiva."
            ),
        ),
        (
            "capacidade_compartilhada",
            "SAB_C + SAB_E e DOM_C + DOM_E",
            "Feiras simultâneas utilizam a mesma capacidade do dia.",
        ),
        (
            "solver",
            SOLVER_METHOD,
            "Dual Simplex da implementação HiGHS do SciPy.",
        ),
        (
            "inteirizacao",
            "maiores restos por grupo de produção",
            (
                "Preserva o total viável sem ultrapassar "
                "previsão ou capacidade."
            ),
        ),
        (
            "carater_configuracao",
            "provisório e revisável",
            (
                "Poderá receber capacidades reais, ingredientes "
                "ou outros recursos."
            ),
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "parametro",
            "valor",
            "justificativa",
        ],
    )


def create_optimization_decision(
    recommendations: pd.DataFrame,
    group_summary: pd.DataFrame,
    source_record_count: int,
    capacity_source: str,
) -> pd.DataFrame:
    """Registra a decisão resultante do snapshot utilizado."""

    active_groups = int(
        group_summary[
            "restricao_capacidade_ativa"
        ].eq("sim").sum()
    )

    return pd.DataFrame(
        [
            {
                "registros_origem_mysql": int(
                    source_record_count
                ),
                "data_corte": recommendations[
                    "data_corte"
                ].max(),
                "fonte_capacidade": capacity_source,
                "algoritmo_solver": SOLVER_METHOD,
                "quantidade_grupos_producao": group_summary[
                    "grupo_producao"
                ].nunique(),
                "grupos_com_capacidade_ativa": active_groups,
                "previsao_total_produtos": int(
                    recommendations[
                        "previsao_produto"
                    ].sum()
                ),
                "recomendacao_total_producao": int(
                    recommendations[
                        "recomendacao_producao"
                    ].sum()
                ),
                "reducao_total_por_capacidade": int(
                    recommendations[
                        "reducao_em_relacao_previsao"
                    ].sum()
                ),
                "recomendacao_final_producao_gerada": "sim",
                "status": (
                    "Solução ótima encontrada e convertida em "
                    "unidades inteiras respeitando previsão, "
                    "mix e capacidades."
                ),
            }
        ]
    )


def validate_production_recommendation(
    recommendations: pd.DataFrame,
    capacities: pd.DataFrame,
    capacity_history: pd.DataFrame,
    group_summary: pd.DataFrame,
    rounding_adjustments: pd.DataFrame,
    solver_diagnostics: pd.DataFrame,
    cutoff_date: pd.Timestamp,
    capacity_source: str,
) -> pd.DataFrame:
    """Audita a capacidade, o modelo linear e a recomendação inteira."""

    checks: list[dict[str, str]] = []

    def add_check(
        name: str,
        passed: bool,
        details: str,
    ) -> None:
        checks.append(
            {
                "teste": name,
                "resultado": "OK" if passed else "FALHA",
                "detalhes": details,
            }
        )

    expected_groups = set(PRODUCTION_GROUP_CONFIGURATION)
    capacity_groups = set(capacities["grupo_producao"])

    add_check(
        "Quatro grupos de produção com capacidade",
        (
            capacity_groups == expected_groups
            and len(capacities) == 4
        ),
        f"Grupos encontrados: {sorted(capacity_groups)}",
    )

    positive_integer_capacities = capacities[
        "capacidade_estimada"
    ].map(
        lambda value: (
            float(value).is_integer()
            and float(value) > 0
        )
    ).all()

    add_check(
        "Capacidades inteiras e positivas",
        bool(positive_integer_capacities),
        "Capacidades: "
        + str(
            capacities.set_index("grupo_producao")[
                "capacidade_estimada"
            ].to_dict()
        ),
    )

    if capacity_source == CAPACITY_SOURCE_HISTORICAL_MAX:
        recalculated_maximum = capacity_history.groupby(
            "grupo_producao"
        )["producao_total_dia"].max()

        informed_capacity = capacities.set_index(
            "grupo_producao"
        )["capacidade_estimada"]

        historical_maximum_ok = all(
            isclose(
                float(informed_capacity.loc[group]),
                float(recalculated_maximum.loc[group]),
                abs_tol=NUMERICAL_TOLERANCE,
            )
            for group in expected_groups
        )
    else:
        historical_maximum_ok = True

    add_check(
        "Capacidade igual ao máximo histórico configurado",
        historical_maximum_ok,
        f"Fonte ativa: {capacity_source}",
    )

    history_before_cutoff = (
        capacity_history["maior_data_venda"]
        <= cutoff_date
    ).all()

    add_check(
        "Capacidade usa somente histórico disponível",
        bool(history_before_cutoff),
        f"Data de corte: {cutoff_date:%Y-%m-%d}",
    )

    add_check(
        "Simplex encontrou solução ótima",
        (
            solver_diagnostics["sucesso"].eq(1).all()
            and solver_diagnostics[
                "status_codigo"
            ].eq(0).all()
        ),
        str(solver_diagnostics["mensagem"].iloc[0]),
    )

    add_check(
        "Método dual Simplex identificado",
        recommendations[
            "algoritmo_solver"
        ].eq(SOLVER_METHOD).all(),
        f"Método esperado: {SOLVER_METHOD}",
    )

    continuous = recommendations["recomendacao_continua"]
    valid_continuous = (
        continuous.map(
            lambda value: isfinite(float(value))
        ).all()
        and (
            continuous >= -NUMERICAL_TOLERANCE
        ).all()
    )

    add_check(
        "Solução contínua válida e não negativa",
        bool(valid_continuous),
        f"Menor valor contínuo: {continuous.min():.10f}",
    )

    continuous_within_forecast = (
        continuous
        <= (
            recommendations["previsao_produto"]
            + NUMERICAL_TOLERANCE
        )
    ).all()

    add_check(
        "Solução contínua não supera a previsão",
        bool(continuous_within_forecast),
        "Cada variável possui a previsão como limite superior.",
    )

    expected_continuous = (
        recommendations["previsao_produto"]
        * recommendations["fator_atendimento_continuo"]
    )
    maximum_mix_difference = float(
        (
            continuous - expected_continuous
        ).abs().max()
    )

    add_check(
        "Mix preservado na solução contínua",
        maximum_mix_difference <= NUMERICAL_TOLERANCE,
        f"Maior diferença: {maximum_mix_difference:.10f}",
    )

    integer_recommendations = recommendations[
        "recomendacao_producao"
    ].map(
        lambda value: float(value).is_integer()
    ).all()

    nonnegative_recommendations = recommendations[
        "recomendacao_producao"
    ].ge(0).all()

    add_check(
        "Recomendações inteiras e não negativas",
        bool(
            integer_recommendations
            and nonnegative_recommendations
        ),
        (
            "Menor recomendação: "
            f"{recommendations['recomendacao_producao'].min()}"
        ),
    )

    recommendations_within_forecast = (
        recommendations["recomendacao_producao"]
        <= recommendations["previsao_produto"]
    ).all()

    add_check(
        "Recomendação não supera a previsão",
        bool(recommendations_within_forecast),
        "Nenhuma produção adicional foi criada além da previsão.",
    )

    capacity_respected = (
        group_summary["recomendacao_total"]
        <= group_summary["capacidade_estimada"]
    ).all()

    add_check(
        "Capacidade respeitada em todos os dias",
        bool(capacity_respected),
        "Maior utilização: "
        f"{group_summary['percentual_utilizacao_capacidade'].max():.4f}%",
    )

    expected_totals = group_summary[
        ["previsao_total", "capacidade_estimada"]
    ].min(axis=1)

    target_totals_ok = group_summary[
        "recomendacao_total"
    ].equals(expected_totals.astype(int))

    add_check(
        "Total recomendado igual ao máximo viável",
        target_totals_ok,
        (
            "O alvo é o menor valor entre previsão "
            "e capacidade de cada grupo."
        ),
    )

    adjustment_values_ok = recommendations[
        "ajuste_maior_resto_recomendacao"
    ].isin([0, 1]).all()

    rounding_totals_ok = rounding_adjustments[
        "soma_final_recomendacao"
    ].equals(
        rounding_adjustments[
            "alvo_inteiro_recomendacao"
        ]
    )

    add_check(
        "Arredondamento reconciliado por maiores restos",
        bool(
            adjustment_values_ok
            and rounding_totals_ok
        ),
        "Cada grupo preserva o total inteiro viável.",
    )

    restricted = recommendations.loc[
        recommendations["produto_permitido"].eq(0)
    ]

    restricted_zero = (
        restricted["previsao_produto"].eq(0).all()
        and restricted[
            "recomendacao_producao"
        ].eq(0).all()
        and set(restricted["id_produto"]) == {5, 15}
        and set(restricted["feira"]) == {
            "SAB_E",
            "DOM_E",
        }
    )

    add_check(
        "Produtos comercialmente restritos permanecem em zero",
        bool(restricted_zero),
        f"Combinações restritas auditadas: {len(restricted)}",
    )

    duplicated = recommendations.duplicated(
        subset=[
            "data_venda_prevista",
            "feira",
            "id_produto",
        ]
    ).sum()

    add_check(
        "Chave futura sem duplicidades",
        duplicated == 0,
        f"Duplicidades encontradas: {int(duplicated)}",
    )

    complete_output = (
        len(recommendations) == EXPECTED_ROWS
        and recommendations["feira"].nunique()
        == EXPECTED_FAIRS
        and recommendations["id_produto"].nunique()
        == EXPECTED_PRODUCTS
    )

    add_check(
        "Cobertura completa de produtos e feiras",
        complete_output,
        (
            f"Linhas: {len(recommendations)}; "
            f"feiras: {recommendations['feira'].nunique()}; "
            "produtos: "
            f"{recommendations['id_produto'].nunique()}"
        ),
    )

    dates_preserved = (
        recommendations[
            "data_producao_prevista"
        ].notna().all()
        and recommendations[
            "data_venda_prevista"
        ].notna().all()
        and (
            recommendations["data_producao_prevista"]
            < recommendations["data_venda_prevista"]
        ).all()
    )

    add_check(
        "Datas operacionais preservadas",
        bool(dates_preserved),
        "A recomendação mantém as datas recebidas da previsão.",
    )

    total_forecast = int(
        recommendations["previsao_produto"].sum()
    )
    total_recommendation = int(
        recommendations["recomendacao_producao"].sum()
    )

    add_check(
        "Total recomendado não supera o total previsto",
        total_recommendation <= total_forecast,
        (
            f"Previsto: {total_forecast}; "
            f"recomendado: {total_recommendation}"
        ),
    )

    source_documented = recommendations[
        "fonte_capacidade"
    ].eq(capacity_source).all()

    add_check(
        "Fonte da capacidade documentada",
        bool(source_documented),
        f"Fonte ativa: {capacity_source}",
    )

    return pd.DataFrame(checks)


def run_production_recommendation(
    source_data: pd.DataFrame,
    product_forecasts: pd.DataFrame,
    source_record_count: int,
    capacity_source: str = DEFAULT_CAPACITY_SOURCE,
    manual_capacities: Mapping[str, int | float] | None = None,
) -> dict[str, pd.DataFrame]:
    """Executa a recomendação de produção da Fase 13."""

    forecasts, cutoff_date = prepare_product_forecasts(
        product_forecasts
    )

    capacities, capacity_history = resolve_capacities(
        source_data=source_data,
        cutoff_date=cutoff_date,
        capacity_source=capacity_source,
        manual_capacities=manual_capacities,
    )

    continuous, solver_diagnostics = (
        solve_linear_recommendation(
            forecasts,
            capacities,
        )
    )

    recommendations, rounding_adjustments = (
        round_recommendations(
            continuous,
            capacities,
        )
    )

    group_summary, fair_summary = (
        create_recommendation_summaries(
            recommendations,
            capacities,
        )
    )

    validation = validate_production_recommendation(
        recommendations=recommendations,
        capacities=capacities,
        capacity_history=capacity_history,
        group_summary=group_summary,
        rounding_adjustments=rounding_adjustments,
        solver_diagnostics=solver_diagnostics,
        cutoff_date=cutoff_date,
        capacity_source=capacity_source,
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
            "Falha na recomendação de produção: "
            + "; ".join(failed)
        )

    output_columns = [
        "data_corte",
        "data_producao_prevista",
        "data_venda_prevista",
        "grupo_producao",
        "dia_producao",
        "feira",
        "categoria",
        "id_produto",
        "produto",
        "produto_permitido",
        "motivo_restricao",
        "previsao_produto",
        "recomendacao_continua",
        "recomendacao_piso",
        "parte_decimal_recomendacao",
        "ajuste_maior_resto_recomendacao",
        "recomendacao_producao",
        "reducao_em_relacao_previsao",
        "percentual_atendimento_produto",
        "fator_atendimento_continuo",
        "capacidade_estimada_grupo",
        "fonte_capacidade",
        "metodo_otimizacao",
        "algoritmo_solver",
    ]

    optional_columns = [
        "modelo_previsao_categoria",
        "nome_modelo_previsao_categoria",
        "participacao_percentual",
    ]

    output_columns.extend(
        column
        for column in optional_columns
        if column in recommendations.columns
    )

    return {
        "recomendacoes_producao_produto_feira": (
            recommendations[output_columns].copy()
        ),
        "capacidades_estimadas_producao": capacities,
        "historico_capacidade_diaria": capacity_history,
        "resumo_recomendacao_por_dia": group_summary,
        "resumo_recomendacao_por_feira": fair_summary,
        "ajustes_arredondamento_recomendacao": (
            rounding_adjustments
        ),
        "diagnostico_solver_simplex": solver_diagnostics,
        "validacao_recomendacao_producao": validation,
        "configuracao_otimizacao_producao": (
            create_optimization_configuration(
                capacity_source
            )
        ),
        "decisao_recomendacao_producao": (
            create_optimization_decision(
                recommendations=recommendations,
                group_summary=group_summary,
                source_record_count=source_record_count,
                capacity_source=capacity_source,
            )
        ),
    }


def save_production_recommendation_outputs(
    analyses: dict[str, pd.DataFrame],
) -> list[Path]:
    """Salva recomendações, capacidades e auditorias da Fase 13."""

    project_root = Path(__file__).resolve().parent.parent
    recommendations_directory = (
        project_root
        / "reports"
        / "recommendations"
    )
    tables_directory = (
        project_root
        / "reports"
        / "tables"
    )

    recommendations_directory.mkdir(
        parents=True,
        exist_ok=True,
    )
    tables_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    recommendation_files = {
        "recomendacoes_producao_produto_feira"
    }
    generated_files: list[Path] = []

    for name, dataframe in analyses.items():
        directory = (
            recommendations_directory
            if name in recommendation_files
            else tables_directory
        )
        path = directory / f"{name}.csv"
        dataframe.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )
        generated_files.append(path)

    return generated_files