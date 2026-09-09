"""
Engenharia de atributos para a previsão de demanda.

A unidade de análise é categoria + feira.

Os atributos históricos são calculados separadamente
dentro de cada série e utilizam somente valores anteriores
à observação que será prevista.

"""

from pathlib import Path

import pandas as pd

from analysis.exploratory import (
    prepare_analysis_data,
)


LAGS = [
    1,
    2,
    3,
]

ROLLING_WINDOWS = [
    2,
    3,
    4,
]

HISTORICAL_FEATURES = [
    "lag_1",
    "lag_2",
    "lag_3",
    "media_movel_2",
    "media_movel_3",
    "media_movel_4",
]


def _validate_operation_context(
    dataframe: pd.DataFrame,
) -> None:
    """
    Garante que cada operação possua
    um único contexto.

    Uma mesma operação não pode ter datas,
    feira, clima ou indicador de feriado
    diferentes entre seus produtos.
    """

    context_columns = [
        "data_producao",
        "data_venda",
        "feira",
        "clima",
        "eh_feriado",
        "nome_feriado",
    ]

    for column in context_columns:
        counts = (
            dataframe.groupby(
                "id_operacao"
            )[column]
            .nunique(
                dropna=False
            )
        )

        if (
            counts > 1
        ).any():
            raise ValueError(
                "Uma mesma operação possui "
                "valores diferentes para a coluna "
                f"{column}."
            )


def create_category_fair_base(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega os registros no nível:

    data de venda + feira + categoria.
    """

    df = prepare_analysis_data(
        dataframe
    )

    _validate_operation_context(
        df
    )

    base = (
        df.groupby(
            [
                "id_operacao",
                "data_venda",
                "feira",
                "categoria",
            ],
            dropna=False,
        )
        .agg(
            data_producao=(
                "data_producao",
                "first",
            ),

            clima=(
                "clima",
                "first",
            ),

            eh_feriado=(
                "eh_feriado",
                "first",
            ),

            nome_feriado=(
                "nome_feriado",
                "first",
            ),

            demanda_observada=(
                "quantidade_vendida",
                "sum",
            ),

            total_produzido=(
                "quantidade_produzida",
                "sum",
            ),

            total_sobra=(
                "quantidade_sobra",
                "sum",
            ),

            quantidade_produtos=(
                "id_produto",
                "nunique",
            ),

            produtos_possivelmente_censurados=(
                "possivel_demanda_censurada",
                "sum",
            ),
        )
        .reset_index()
    )

    base["serie"] = (
        base["feira"].astype(str)
        + " | "
        + base["categoria"].astype(str)
    )

    base[
        "percentual_produtos_possivelmente_censurados"
    ] = (
        base[
            "produtos_possivelmente_censurados"
        ]
        / base[
            "quantidade_produtos"
        ]
        * 100
    ).round(2)

    base = base.sort_values(
        [
            "serie",
            "data_venda",
        ]
    ).reset_index(
        drop=True
    )

    base[
        "numero_observacao_serie"
    ] = (
        base.groupby(
            "serie"
        ).cumcount()
        + 1
    )

    return base


def add_temporal_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria atributos temporais conhecidos
    antes da realização da feira.
    """

    df = dataframe.copy()

    dates = pd.to_datetime(
        df["data_venda"]
    )

    iso_calendar = (
        dates.dt.isocalendar()
    )

    df["ano"] = (
        dates.dt.year
    )

    df["mes"] = (
        dates.dt.month
    )

    df["trimestre"] = (
        dates.dt.quarter
    )

    df["semana_ano"] = (
        iso_calendar.week.astype(int)
    )

    df["dia_semana"] = (
        dates.dt.dayofweek
    )

    df["fim_de_semana"] = (
        df["dia_semana"] >= 5
    ).astype(int)

    df["tendencia_serie"] = (
        df.groupby(
            "serie"
        ).cumcount()
        + 1
    )

    return df


def add_historical_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria lags e médias móveis sem
    vazamento temporal.

    Todos os cálculos são realizados
    dentro da mesma série.

    O shift(1) garante que a demanda da
    própria linha não seja utilizada.
    """

    df = dataframe.copy()

    target_by_series = (
        df.groupby(
            "serie"
        )["demanda_observada"]
    )

    for lag in LAGS:
        df[
            f"lag_{lag}"
        ] = target_by_series.shift(
            lag
        )

    for window in ROLLING_WINDOWS:
        df[
            f"media_movel_{window}"
        ] = (
            df.groupby(
                "serie"
            )["demanda_observada"]
            .transform(
                lambda values: (
                    values.shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            )
            .round(2)
        )

    return df


def create_modeling_dataset(
    feature_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Remove somente as linhas iniciais
    sem histórico suficiente.

    A janela máxima é de quatro
    observações anteriores.
    """

    modeling_data = (
        feature_data.dropna(
            subset=HISTORICAL_FEATURES
        )
        .reset_index(
            drop=True
        )
    )

    if modeling_data.empty:
        raise ValueError(
            "A criação dos atributos históricos "
            "removeu todas as linhas do dataset."
        )

    return modeling_data


def create_loss_by_series(
    feature_data: pd.DataFrame,
    modeling_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Quantifica a perda de linhas causada
    pelas janelas históricas.
    """

    before = (
        feature_data.groupby(
            [
                "serie",
                "feira",
                "categoria",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "linhas_antes"
        )
        .reset_index()
    )

    after = (
        modeling_data.groupby(
            [
                "serie",
                "feira",
                "categoria",
            ],
            dropna=False,
        )
        .size()
        .rename(
            "linhas_depois"
        )
        .reset_index()
    )

    result = before.merge(
        after,
        on=[
            "serie",
            "feira",
            "categoria",
        ],
        how="left",
    )

    result["linhas_depois"] = (
        result[
            "linhas_depois"
        ]
        .fillna(0)
        .astype(int)
    )

    result["linhas_removidas"] = (
        result["linhas_antes"]
        - result["linhas_depois"]
    )

    result["percentual_mantido"] = (
        result["linhas_depois"]
        / result["linhas_antes"]
        * 100
    ).round(2)

    return result.sort_values(
        "serie"
    ).reset_index(
        drop=True
    )


def _same_values(
    left: pd.Series,
    right: pd.Series,
) -> bool:
    """
    Compara duas séries considerando
    valores ausentes equivalentes.
    """

    sentinel = (
        -999999999.0
    )

    return (
        left.fillna(
            sentinel
        ).equals(
            right.fillna(
                sentinel
            )
        )
    )


def validate_engineered_features(
    feature_data: pd.DataFrame,
    modeling_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Valida:

    - chaves;
    - ordenação temporal;
    - separação entre séries;
    - lags;
    - médias móveis;
    - valores ausentes;
    - preservação das 18 séries.
    """

    checks: list[
        dict[str, str]
    ] = []

    def add_check(
        name: str,
        passed: bool,
        details: str,
    ) -> None:
        checks.append(
            {
                "teste": name,

                "resultado": (
                    "OK"
                    if passed
                    else "FALHA"
                ),

                "detalhes": details,
            }
        )

    # ========================================================
    # CHAVE ÚNICA
    # ========================================================

    duplicated = (
        feature_data.duplicated(
            subset=[
                "data_venda",
                "feira",
                "categoria",
            ]
        ).sum()
    )

    add_check(
        name=(
            "Chave data + feira + categoria"
        ),
        passed=(
            duplicated == 0
        ),
        details=(
            "Duplicidades encontradas: "
            f"{int(duplicated)}"
        ),
    )

    # ========================================================
    # ORDEM TEMPORAL
    # ========================================================

    ordered = all(
        group[
            "data_venda"
        ].is_monotonic_increasing
        for _, group in (
            feature_data.groupby(
                "serie"
            )
        )
    )

    add_check(
        name=(
            "Ordenação temporal por série"
        ),
        passed=ordered,
        details=(
            "As datas devem estar em ordem "
            "crescente dentro de cada série."
        ),
    )

    # ========================================================
    # LAGS
    # ========================================================

    for lag in LAGS:
        expected = (
            feature_data.groupby(
                "serie"
            )["demanda_observada"]
            .shift(
                lag
            )
        )

        add_check(
            name=(
                f"lag_{lag} sem vazamento"
            ),
            passed=_same_values(
                feature_data[
                    f"lag_{lag}"
                ],
                expected,
            ),
            details=(
                f"lag_{lag} deve usar somente "
                "valores anteriores da mesma série."
            ),
        )

    # ========================================================
    # MÉDIAS MÓVEIS
    # ========================================================

    for window in ROLLING_WINDOWS:
        expected = (
            feature_data.groupby(
                "serie"
            )["demanda_observada"]
            .transform(
                lambda values: (
                    values.shift(1)
                    .rolling(
                        window=window,
                        min_periods=window,
                    )
                    .mean()
                )
            )
            .round(2)
        )

        add_check(
            name=(
                f"media_movel_{window} "
                "sem vazamento"
            ),
            passed=_same_values(
                feature_data[
                    f"media_movel_{window}"
                ],
                expected,
            ),
            details=(
                "A média deve excluir a demanda "
                "da própria linha."
            ),
        )

    # ========================================================
    # VALORES AUSENTES
    # ========================================================

    missing = (
        modeling_data[
            HISTORICAL_FEATURES
        ]
        .isna()
        .sum()
        .sum()
    )

    add_check(
        name=(
            "Dataset de modelagem sem "
            "nulos históricos"
        ),
        passed=(
            missing == 0
        ),
        details=(
            "Valores ausentes encontrados: "
            f"{int(missing)}"
        ),
    )

    # ========================================================
    # SÉRIES
    # ========================================================

    series_count = (
        modeling_data[
            "serie"
        ].nunique()
    )

    add_check(
        name=(
            "Preservação das 18 séries"
        ),
        passed=(
            series_count == 18
        ),
        details=(
            f"Séries presentes: "
            f"{series_count}"
        ),
    )

    return pd.DataFrame(
        checks
    )


def create_feature_summary(
    source_data: pd.DataFrame,
    feature_data: pd.DataFrame,
    modeling_data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria o resumo quantitativo
    da engenharia de atributos.
    """

    original_rows = len(
        feature_data
    )

    modeling_rows = len(
        modeling_data
    )

    removed_rows = (
        original_rows
        - modeling_rows
    )

    observations_by_series = (
        feature_data.groupby(
            "serie"
        ).size()
    )

    modeling_observations_by_series = (
        modeling_data.groupby(
            "serie"
        ).size()
    )

    censored_items = (
        feature_data[
            "produtos_possivelmente_censurados"
        ].sum()
    )

    total_items = (
        feature_data[
            "quantidade_produtos"
        ].sum()
    )

    lines_with_censored_items = (
        feature_data[
            "produtos_possivelmente_censurados"
        ]
        > 0
    )

    summary = {
        "registros_origem_mysql": (
            len(source_data)
        ),

        "linhas_categoria_feira": (
            original_rows
        ),

        "quantidade_series": (
            feature_data[
                "serie"
            ].nunique()
        ),

        "observacoes_minimas_por_serie": int(
            observations_by_series.min()
        ),

        "observacoes_maximas_por_serie": int(
            observations_by_series.max()
        ),

        "janela_historica_maxima": (
            max(
                ROLLING_WINDOWS
            )
        ),

        "linhas_removidas_por_historico": (
            removed_rows
        ),

        "linhas_dataset_modelagem": (
            modeling_rows
        ),

        "percentual_dados_mantidos": round(
            modeling_rows
            / original_rows
            * 100,
            2,
        ),

        "observacoes_minimas_modelagem_por_serie": int(
            modeling_observations_by_series.min()
        ),

        "observacoes_maximas_modelagem_por_serie": int(
            modeling_observations_by_series.max()
        ),

        "itens_possivelmente_censurados": int(
            censored_items
        ),

        "percentual_itens_possivelmente_censurados": round(
            censored_items
            / total_items
            * 100,
            2,
        ),

        "linhas_com_algum_item_possivelmente_censurado": int(
            lines_with_censored_items.sum()
        ),

        "percentual_linhas_com_algum_item_censurado": round(
            lines_with_censored_items.mean()
            * 100,
            2,
        ),
    }

    return pd.DataFrame(
        [summary]
    )


def create_feature_dictionary() -> pd.DataFrame:
    """
    Documenta os atributos e sua
    disponibilidade no momento da previsão.
    """

    rows = [
        (
            "data_venda",
            "metadado",
            "Data da feira a ser prevista.",
        ),

        (
            "serie",
            "identificador",
            "Combinação entre feira e categoria.",
        ),

        (
            "feira",
            "categórico",
            "Feira conhecida antes da previsão.",
        ),

        (
            "categoria",
            "categórico",
            "Categoria conhecida antes da previsão.",
        ),

        (
            "clima",
            "categórico",
            "Clima observado ou informado "
            "para a previsão futura.",
        ),

        (
            "eh_feriado",
            "binário",
            "Indica feriado conhecido antecipadamente.",
        ),

        (
            "ano",
            "temporal",
            "Ano da data de venda.",
        ),

        (
            "mes",
            "temporal",
            "Mês da data de venda.",
        ),

        (
            "trimestre",
            "temporal",
            "Trimestre da data de venda.",
        ),

        (
            "semana_ano",
            "temporal",
            "Semana ISO do ano.",
        ),

        (
            "dia_semana",
            "temporal",
            "Dia da semana, de 0 a 6.",
        ),

        (
            "fim_de_semana",
            "binário",
            "Indica sábado ou domingo.",
        ),

        (
            "tendencia_serie",
            "temporal",
            "Posição cronológica dentro da série.",
        ),

        (
            "lag_1",
            "histórico",
            "Demanda da ocorrência anterior "
            "da mesma série.",
        ),

        (
            "lag_2",
            "histórico",
            "Demanda de duas ocorrências anteriores.",
        ),

        (
            "lag_3",
            "histórico",
            "Demanda de três ocorrências anteriores.",
        ),

        (
            "media_movel_2",
            "histórico",
            "Média das duas ocorrências anteriores.",
        ),

        (
            "media_movel_3",
            "histórico",
            "Média das três ocorrências anteriores.",
        ),

        (
            "media_movel_4",
            "histórico",
            "Média das quatro ocorrências anteriores.",
        ),

        (
            "demanda_observada",
            "alvo",
            "Quantidade vendida agregada "
            "por categoria e feira.",
        ),

        (
            "percentual_produtos_possivelmente_censurados",
            "auditoria",
            "Indicador posterior à venda; "
            "não deve ser usado como preditor.",
        ),

        (
            "total_produzido / total_sobra",
            "auditoria",
            "Resultados da operação; "
            "não devem ser usados como preditores.",
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=[
            "atributo",
            "tipo",
            "descricao",
        ],
    )


def run_feature_engineering(
    dataframe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:

    category_fair = (
        create_category_fair_base(
            dataframe
        )
    )

    with_temporal = (
        add_temporal_features(
            category_fair
        )
    )

    feature_data = (
        add_historical_features(
            with_temporal
        )
    )

    modeling_data = (
        create_modeling_dataset(
            feature_data
        )
    )

    loss_by_series = (
        create_loss_by_series(
            feature_data=feature_data,
            modeling_data=modeling_data,
        )
    )

    validation = (
        validate_engineered_features(
            feature_data=feature_data,
            modeling_data=modeling_data,
        )
    )

    if (
        validation[
            "resultado"
        ] == "FALHA"
    ).any():
        failed = validation.loc[
            validation[
                "resultado"
            ] == "FALHA",
            "teste",
        ].tolist()

        raise ValueError(
            "Falha na validação da engenharia "
            "de atributos: "
            + ", ".join(
                failed
            )
        )

    summary = (
        create_feature_summary(
            source_data=dataframe,
            feature_data=feature_data,
            modeling_data=modeling_data,
        )
    )

    return {
        "base_categoria_feira_com_atributos": (
            feature_data
        ),

        "dataset_modelagem_categoria_feira": (
            modeling_data
        ),

        "resumo_engenharia_atributos": (
            summary
        ),

        "validacao_engenharia_atributos": (
            validation
        ),

        "perda_linhas_por_serie": (
            loss_by_series
        ),

        "dicionario_atributos": (
            create_feature_dictionary()
        ),
    }


def save_feature_engineering_outputs(
    analyses: dict[str, pd.DataFrame],
) -> list[Path]:

    project_root = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    processed_directory = (
        project_root
        / "data"
        / "processed"
    )

    tables_directory = (
        project_root
        / "reports"
        / "tables"
    )

    processed_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    tables_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    destinations = {
        "base_categoria_feira_com_atributos": (
            processed_directory
        ),

        "dataset_modelagem_categoria_feira": (
            processed_directory
        ),

        "resumo_engenharia_atributos": (
            tables_directory
        ),

        "validacao_engenharia_atributos": (
            tables_directory
        ),

        "perda_linhas_por_serie": (
            tables_directory
        ),

        "dicionario_atributos": (
            tables_directory
        ),
    }

    generated_files: list[
        Path
    ] = []

    for name, dataframe in (
        analyses.items()
    ):
        path = (
            destinations[name]
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