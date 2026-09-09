"""
Análise da granularidade da previsão de demanda.

Esta etapa compara quatro níveis de agregação:

- produto + feira;
- categoria + feira;
- feira;
- total do negócio por data de venda.

Nenhum modelo de previsão é treinado neste módulo.
"""

from pathlib import Path

import pandas as pd

from analysis.exploratory import prepare_analysis_data


GRANULARITY_ORDER = [
    "produto_feira",
    "categoria_feira",
    "feira",
    "total",
]

GRANULARITY_NAMES = {
    "produto_feira": "Produto + feira",
    "categoria_feira": "Categoria + feira",
    "feira": "Feira",
    "total": "Total geral",
}

MINIMUM_REFERENCE = 12


def prepare_granularity_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara e valida os dados usados na análise.
    """

    df = prepare_analysis_data(
        dataframe
    )

    df["id_produto"] = pd.to_numeric(
        df["id_produto"],
        errors="raise",
    )

    # Frango (5) e Escarola sem bacon (15)
    # não são vendidos em SAB_E e DOM_E.
    restricted = (
        df["feira"].isin(
            [
                "SAB_E",
                "DOM_E",
            ]
        )
        & df["id_produto"].isin(
            [
                5,
                15,
            ]
        )
        & (
            (
                df["quantidade_produzida"]
                > 0
            )
            | (
                df["quantidade_vendida"]
                > 0
            )
            | (
                df["quantidade_sobra"]
                > 0
            )
        )
    )

    if restricted.any():
        raise ValueError(
            "Foram encontrados valores positivos para "
            "produtos que não são vendidos nas feiras "
            "SAB_E ou DOM_E."
        )

    return df


def _build_granularity_dataset(
    dataframe: pd.DataFrame,
    granularity: str,
) -> pd.DataFrame:
    """
    Cria a série temporal correspondente
    a uma granularidade.
    """

    configurations = {
        "produto_feira": [
            "feira",
            "id_produto",
            "produto",
            "categoria",
        ],

        "categoria_feira": [
            "feira",
            "categoria",
        ],

        "feira": [
            "feira",
        ],

        "total": [],
    }

    if granularity not in configurations:
        raise ValueError(
            f"Granularidade inválida: "
            f"{granularity}"
        )

    series_columns = configurations[
        granularity
    ]

    group_columns = [
        "data_venda",
        *series_columns,
    ]

    result = (
        dataframe.groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            demanda_observada=(
                "quantidade_vendida",
                "sum",
            ),

            quantidade_operacoes=(
                "id_operacao",
                "nunique",
            ),
        )
        .reset_index()
    )

    if granularity == "produto_feira":
        result["serie"] = (
            result["feira"].astype(str)
            + " | "
            + result["id_produto"].astype(str)
            + " - "
            + result["produto"].astype(str)
        )

    elif granularity == "categoria_feira":
        result["serie"] = (
            result["feira"].astype(str)
            + " | "
            + result["categoria"].astype(str)
        )

    elif granularity == "feira":
        result["serie"] = (
            result["feira"].astype(str)
        )

    else:
        result["serie"] = (
            "TOTAL_GERAL"
        )

    result["granularidade"] = (
        granularity
    )

    result = result.sort_values(
        [
            "serie",
            "data_venda",
        ]
    ).reset_index(
        drop=True
    )

    result["numero_observacao"] = (
        result.groupby(
            "serie"
        ).cumcount()
        + 1
    )

    return result


def _classify_history(
    observations: int,
) -> str:
    """
    Classifica o tamanho do histórico.

    A classificação não garante que o histórico
    seja suficiente para determinado modelo.
    """

    if observations < 8:
        return "Muito curto"

    if observations < 12:
        return "Curto"

    if observations < 24:
        return "Limitado"

    return (
        "Mais adequado para "
        "avaliação temporal"
    )


def _build_series_profile(
    dataset: pd.DataFrame,
    granularity: str,
) -> pd.DataFrame:
    """
    Calcula o perfil estatístico
    de cada série temporal.
    """

    profile = (
        dataset.groupby(
            "serie",
            dropna=False,
        )
        .agg(
            quantidade_observacoes=(
                "data_venda",
                "size",
            ),

            quantidade_datas_unicas=(
                "data_venda",
                "nunique",
            ),

            primeira_data=(
                "data_venda",
                "min",
            ),

            ultima_data=(
                "data_venda",
                "max",
            ),

            demanda_total=(
                "demanda_observada",
                "sum",
            ),

            demanda_media=(
                "demanda_observada",
                "mean",
            ),

            demanda_mediana=(
                "demanda_observada",
                "median",
            ),

            desvio_padrao_demanda=(
                "demanda_observada",
                "std",
            ),

            quantidade_demanda_zero=(
                "demanda_observada",
                lambda values: int(
                    (
                        values == 0
                    ).sum()
                ),
            ),
        )
        .reset_index()
    )

    profile["granularidade"] = (
        granularity
    )

    profile["nome_granularidade"] = (
        GRANULARITY_NAMES[
            granularity
        ]
    )

    profile[
        "periodo_calendario_dias"
    ] = (
        profile["ultima_data"]
        - profile["primeira_data"]
    ).dt.days + 1

    profile[
        "percentual_demanda_zero"
    ] = (
        profile[
            "quantidade_demanda_zero"
        ]
        / profile[
            "quantidade_observacoes"
        ]
        * 100
    ).round(2)

    profile[
        "desvio_padrao_demanda"
    ] = (
        profile[
            "desvio_padrao_demanda"
        ]
        .fillna(0)
        .round(2)
    )

    profile["demanda_media"] = (
        profile[
            "demanda_media"
        ].round(2)
    )

    profile["demanda_mediana"] = (
        profile[
            "demanda_mediana"
        ].round(2)
    )

    valid_mean = profile[
        "demanda_media"
    ].where(
        profile[
            "demanda_media"
        ] != 0
    )

    profile[
        "coeficiente_variacao"
    ] = (
        profile[
            "desvio_padrao_demanda"
        ]
        / valid_mean
    ).round(4)

    profile[
        "atinge_referencia_12"
    ] = (
        profile[
            "quantidade_observacoes"
        ]
        >= MINIMUM_REFERENCE
    )

    profile[
        "classificacao_historico"
    ] = (
        profile[
            "quantidade_observacoes"
        ].apply(
            _classify_history
        )
    )

    columns = [
        "granularidade",
        "nome_granularidade",
        "serie",
        "quantidade_observacoes",
        "quantidade_datas_unicas",
        "primeira_data",
        "ultima_data",
        "periodo_calendario_dias",
        "demanda_total",
        "demanda_media",
        "demanda_mediana",
        "desvio_padrao_demanda",
        "coeficiente_variacao",
        "quantidade_demanda_zero",
        "percentual_demanda_zero",
        "atinge_referencia_12",
        "classificacao_historico",
    ]

    return profile[
        columns
    ].sort_values(
        [
            "granularidade",
            "serie",
        ]
    ).reset_index(
        drop=True
    )


def _percentage_at_least(
    profile: pd.DataFrame,
    minimum: int,
) -> float:
    """
    Calcula o percentual de séries
    com o mínimo informado.
    """

    if profile.empty:
        return 0.0

    return round(
        (
            profile[
                "quantidade_observacoes"
            ]
            >= minimum
        ).mean()
        * 100,
        2,
    )


def _build_granularity_summary(
    profiles: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """
    Resume e compara as granularidades.
    """

    rows = []

    for granularity in (
        GRANULARITY_ORDER
    ):
        profile = profiles[
            granularity
        ]

        observations = profile[
            "quantidade_observacoes"
        ]

        rows.append(
            {
                "granularidade": (
                    granularity
                ),

                "nome_granularidade": (
                    GRANULARITY_NAMES[
                        granularity
                    ]
                ),

                "quantidade_series": (
                    len(profile)
                ),

                "total_observacoes": int(
                    observations.sum()
                ),

                "minimo_observacoes_por_serie": int(
                    observations.min()
                ),

                "mediana_observacoes_por_serie": round(
                    float(
                        observations.median()
                    ),
                    2,
                ),

                "media_observacoes_por_serie": round(
                    float(
                        observations.mean()
                    ),
                    2,
                ),

                "maximo_observacoes_por_serie": int(
                    observations.max()
                ),

                "percentual_series_com_8_ou_mais": (
                    _percentage_at_least(
                        profile,
                        8,
                    )
                ),

                "percentual_series_com_12_ou_mais": (
                    _percentage_at_least(
                        profile,
                        12,
                    )
                ),

                "percentual_series_com_16_ou_mais": (
                    _percentage_at_least(
                        profile,
                        16,
                    )
                ),

                "percentual_series_com_24_ou_mais": (
                    _percentage_at_least(
                        profile,
                        24,
                    )
                ),

                "media_percentual_demanda_zero": round(
                    float(
                        profile[
                            "percentual_demanda_zero"
                        ].mean()
                    ),
                    2,
                ),

                "mediana_coeficiente_variacao": round(
                    float(
                        profile[
                            "coeficiente_variacao"
                        ].median()
                    ),
                    4,
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def create_granularity_decision(
    summary: pd.DataFrame,
) -> pd.DataFrame:
    """
    Documenta a granularidade escolhida
    para a previsão.
    """

    selected = summary.loc[
        summary["granularidade"]
        == "categoria_feira"
    ].iloc[0]

    decision = {
        "granularidade_escolhida": (
            "categoria_feira"
        ),

        "nome_granularidade": (
            "Categoria + feira"
        ),

        "unidade_da_serie": (
            "feira + categoria"
        ),

        "variavel_alvo": (
            "demanda observada "
            "(quantidade_vendida)"
        ),

        "quantidade_series": int(
            selected[
                "quantidade_series"
            ]
        ),

        "total_observacoes": int(
            selected[
                "total_observacoes"
            ]
        ),

        "observacoes_mediana_por_serie": float(
            selected[
                "mediana_observacoes_por_serie"
            ]
        ),

        "percentual_medio_demanda_zero": float(
            selected[
                "media_percentual_demanda_zero"
            ]
        ),

        "coeficiente_variacao_mediano": float(
            selected[
                "mediana_coeficiente_variacao"
            ]
        ),

        "estrategia_de_modelagem": (
            "Modelagem global utilizando feira e "
            "categoria como atributos, mantendo "
            "lags separados por série."
        ),

        "estrategia_para_produtos": (
            "Distribuir posteriormente a previsão "
            "da categoria entre os produtos utilizando "
            "proporções históricas e as restrições "
            "comerciais."
        ),

        "justificativa": (
            "A granularidade categoria + feira oferece "
            "o melhor equilíbrio entre quantidade de "
            "séries, estabilidade, detalhamento "
            "operacional e histórico disponível."
        ),
    }

    return pd.DataFrame(
        [decision]
    )


def run_granularity_analysis(
    dataframe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Executa toda a análise quantitativa
    da granularidade.
    """

    df = prepare_granularity_data(
        dataframe
    )

    datasets: dict[
        str,
        pd.DataFrame,
    ] = {}

    profiles: dict[
        str,
        pd.DataFrame,
    ] = {}

    for granularity in (
        GRANULARITY_ORDER
    ):
        dataset = (
            _build_granularity_dataset(
                dataframe=df,
                granularity=granularity,
            )
        )

        datasets[
            granularity
        ] = dataset

        profiles[
            granularity
        ] = _build_series_profile(
            dataset=dataset,
            granularity=granularity,
        )

    combined_profile = pd.concat(
        [
            profiles[item]
            for item in GRANULARITY_ORDER
        ],
        ignore_index=True,
    )

    summary = (
        _build_granularity_summary(
            profiles
        )
    )

    decision = (
        create_granularity_decision(
            summary
        )
    )

    return {
        "resumo_granularidades": (
            summary
        ),

        "decisao_granularidade": (
            decision
        ),

        "perfil_series_granularidade": (
            combined_profile
        ),

        "dados_produto_feira": (
            datasets[
                "produto_feira"
            ]
        ),

        "dados_categoria_feira": (
            datasets[
                "categoria_feira"
            ]
        ),

        "dados_feira": (
            datasets[
                "feira"
            ]
        ),

        "dados_total": (
            datasets[
                "total"
            ]
        ),
    }


def save_granularity_tables(
    analyses: dict[
        str,
        pd.DataFrame,
    ],
) -> list[Path]:
    """
    Salva as tabelas da Fase 5 em CSV.
    """

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    output_directory = (
        project_root
        / "reports"
        / "tables"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    generated_files: list[
        Path
    ] = []

    for name, dataframe in (
        analyses.items()
    ):
        path = (
            output_directory
            / f"{name}.csv"
        )

        dataframe.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        generated_files.append(
            path
        )

    return generated_files