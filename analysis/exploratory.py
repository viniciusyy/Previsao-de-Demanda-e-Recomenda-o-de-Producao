"""
Análise exploratória dos dados históricos da pastelaria.

Este módulo é responsável por calcular estatísticas descritivas
da base utilizada no TCC.

IMPORTANTE:

As análises desta etapa são descritivas.

Diferenças observadas entre clima, feriados, feiras ou outros
atributos não devem ser interpretadas automaticamente como
relações de causa e efeito.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# COLUNAS NUMÉRICAS PRINCIPAIS
# ============================================================

QUANTITY_COLUMNS = [
    "quantidade_produzida",
    "quantidade_vendida",
    "quantidade_sobra",
]


def prepare_analysis_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepara uma cópia dos dados para análise exploratória.

    Nenhuma informação do banco é alterada.

    Args:
        dataframe:
            Dados históricos carregados do MySQL.

    Returns:
        pd.DataFrame:
            Cópia preparada para análise.
    """

    if dataframe.empty:
        raise ValueError(
            "Não é possível realizar análise exploratória "
            "com uma base vazia."
        )

    df = dataframe.copy()

    # ========================================================
    # DATAS
    # ========================================================

    df["data_producao"] = pd.to_datetime(
        df["data_producao"]
    )

    df["data_venda"] = pd.to_datetime(
        df["data_venda"]
    )

    # ========================================================
    # QUANTIDADES
    # ========================================================

    for column in QUANTITY_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        )

    # ========================================================
    # FERIADO
    # ========================================================

    df["eh_feriado"] = pd.to_numeric(
        df["eh_feriado"],
        errors="raise",
    )

    # ========================================================
    # POSSÍVEL DEMANDA CENSURADA
    # ========================================================
    #
    # A identificação é realizada por:
    #
    # produto + operação
    #
    # e não para a feira inteira.
    # ========================================================

    df["possivel_demanda_censurada"] = (
        (df["quantidade_produzida"] > 0)
        & (df["quantidade_sobra"] == 0)
        & (
            df["quantidade_vendida"]
            == df["quantidade_produzida"]
        )
    )

    return df


def create_operation_level_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Converte os registros por produto em dados agregados por operação.

    Uma operação representa uma combinação específica de:

    - id_operacao;
    - data de venda;
    - feira;
    - clima;
    - indicador de feriado.

    Essa agregação é especialmente importante para análises
    de clima e feriados.

    Isso evita comparar incorretamente quantidade de registros
    de produtos com quantidade de operações reais.

    Returns:
        pd.DataFrame:
            Uma linha para cada operação.
    """

    df = prepare_analysis_data(
        dataframe
    )

    operation_data = (
        df.groupby(
            [
                "id_operacao",
                "data_producao",
                "data_venda",
                "feira",
                "clima",
                "eh_feriado",
            ],
            dropna=False,
        )
        .agg(
            total_produzido=(
                "quantidade_produzida",
                "sum",
            ),
            total_vendido=(
                "quantidade_vendida",
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

    operation_data["taxa_sobra_percentual"] = (
        operation_data.apply(
            lambda row: calculate_leftover_rate(
                leftover=row["total_sobra"],
                produced=row["total_produzido"],
            ),
            axis=1,
        )
    )

    return operation_data


def calculate_leftover_rate(
    leftover: float,
    produced: float,
) -> float:
    """
    Calcula a taxa percentual de sobra.
    """

    if produced == 0:
        return 0.0

    return (
        leftover
        / produced
        * 100
    )


def calculate_censored_rate(
    censored_count: int,
    record_count: int,
) -> float:
    """
    Calcula o percentual de registros classificados como
    possível demanda censurada.
    """

    if record_count == 0:
        return 0.0

    return (
        censored_count
        / record_count
        * 100
    )


def create_general_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Cria o resumo geral da base histórica.
    """

    df = prepare_analysis_data(
        dataframe
    )

    first_date = df[
        "data_venda"
    ].min()

    last_date = df[
        "data_venda"
    ].max()

    total_produced = df[
        "quantidade_produzida"
    ].sum()

    total_sold = df[
        "quantidade_vendida"
    ].sum()

    total_leftover = df[
        "quantidade_sobra"
    ].sum()

    leftover_rate = calculate_leftover_rate(
        leftover=total_leftover,
        produced=total_produced,
    )

    censored_count = int(
        df[
            "possivel_demanda_censurada"
        ].sum()
    )

    censored_rate = calculate_censored_rate(
        censored_count=censored_count,
        record_count=len(df),
    )

    period_days = (
        last_date
        - first_date
    ).days + 1

    summary = {
        "quantidade_registros": len(df),

        "quantidade_operacoes": (
            df["id_operacao"].nunique()
        ),

        "quantidade_feiras": (
            df["feira"].nunique()
        ),

        "quantidade_produtos": (
            df["id_produto"].nunique()
        ),

        "quantidade_categorias": (
            df["categoria"].nunique()
        ),

        "primeira_data_venda": (
            first_date.date()
        ),

        "ultima_data_venda": (
            last_date.date()
        ),

        "periodo_calendario_dias": (
            period_days
        ),

        "total_produzido": (
            total_produced
        ),

        "total_vendido": (
            total_sold
        ),

        "total_sobra": (
            total_leftover
        ),

        "taxa_sobra_percentual": (
            round(
                leftover_rate,
                2,
            )
        ),

        "possivel_demanda_censurada": (
            censored_count
        ),

        "percentual_possivel_demanda_censurada": (
            round(
                censored_rate,
                2,
            )
        ),
    }

    return pd.DataFrame(
        [summary]
    )


def _create_group_analysis(
    dataframe: pd.DataFrame,
    group_columns: list[str],
) -> pd.DataFrame:
    """
    Cria análise agregada baseada em registros por produto.

    Esta função é apropriada para:

    - feira;
    - categoria;
    - produto;
    - data;
    - data + feira.

    Para clima e feriados é utilizada análise específica
    em nível de operação.
    """

    df = dataframe.copy()

    grouped = (
        df.groupby(
            group_columns,
            dropna=False,
        )
        .agg(
            quantidade_registros=(
                "id_operacao",
                "size",
            ),

            quantidade_operacoes=(
                "id_operacao",
                "nunique",
            ),

            total_produzido=(
                "quantidade_produzida",
                "sum",
            ),

            total_vendido=(
                "quantidade_vendida",
                "sum",
            ),

            total_sobra=(
                "quantidade_sobra",
                "sum",
            ),

            media_vendida_por_registro=(
                "quantidade_vendida",
                "mean",
            ),

            mediana_vendida_por_registro=(
                "quantidade_vendida",
                "median",
            ),

            possivel_demanda_censurada=(
                "possivel_demanda_censurada",
                "sum",
            ),
        )
        .reset_index()
    )

    grouped[
        "taxa_sobra_percentual"
    ] = grouped.apply(
        lambda row: calculate_leftover_rate(
            leftover=row["total_sobra"],
            produced=row["total_produzido"],
        ),
        axis=1,
    )

    grouped[
        "percentual_possivel_demanda_censurada"
    ] = grouped.apply(
        lambda row: calculate_censored_rate(
            censored_count=int(
                row[
                    "possivel_demanda_censurada"
                ]
            ),
            record_count=int(
                row[
                    "quantidade_registros"
                ]
            ),
        ),
        axis=1,
    )

    grouped[
        "media_vendida_por_registro"
    ] = (
        grouped[
            "media_vendida_por_registro"
        ].round(2)
    )

    grouped[
        "mediana_vendida_por_registro"
    ] = (
        grouped[
            "mediana_vendida_por_registro"
        ].round(2)
    )

    grouped[
        "taxa_sobra_percentual"
    ] = (
        grouped[
            "taxa_sobra_percentual"
        ].round(2)
    )

    grouped[
        "percentual_possivel_demanda_censurada"
    ] = (
        grouped[
            "percentual_possivel_demanda_censurada"
        ].round(2)
    )

    return grouped


def analyze_by_fair(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analisa produção, venda e sobra por feira.
    """

    df = prepare_analysis_data(
        dataframe
    )

    result = _create_group_analysis(
        df,
        ["feira"],
    )

    # ========================================================
    # MÉDIAS POR OPERAÇÃO
    # ========================================================

    result[
        "media_produzida_por_operacao"
    ] = (
        result["total_produzido"]
        / result["quantidade_operacoes"]
    ).round(2)

    result[
        "media_vendida_por_operacao"
    ] = (
        result["total_vendido"]
        / result["quantidade_operacoes"]
    ).round(2)

    result[
        "media_sobra_por_operacao"
    ] = (
        result["total_sobra"]
        / result["quantidade_operacoes"]
    ).round(2)

    return result.sort_values(
        "feira"
    ).reset_index(
        drop=True
    )


def analyze_by_category(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analisa os dados por categoria.
    """

    df = prepare_analysis_data(
        dataframe
    )

    result = _create_group_analysis(
        df,
        ["categoria"],
    )

    return result.sort_values(
        "total_vendido",
        ascending=False,
    ).reset_index(
        drop=True
    )


def analyze_by_product(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analisa os dados por produto.
    """

    df = prepare_analysis_data(
        dataframe
    )

    result = _create_group_analysis(
        df,
        [
            "id_produto",
            "produto",
            "categoria",
        ],
    )

    return result.sort_values(
        "total_vendido",
        ascending=False,
    ).reset_index(
        drop=True
    )


def analyze_by_climate(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analisa vendas por clima em nível de operação.

    Essa abordagem é mais adequada do que simplesmente
    comparar vendas totais por clima.

    Exemplo:

    50 operações com SOL naturalmente podem produzir um total
    vendido maior que 2 operações com CHUVA_FORTE.

    Por isso, além dos totais, são calculadas:

    - média vendida por operação;
    - mediana vendida por operação;
    - média produzida por operação;
    - média de sobra por operação.

    A análise permanece descritiva e não demonstra causalidade.
    """

    operations = create_operation_level_data(
        dataframe
    )

    result = (
        operations.groupby(
            "clima",
            dropna=False,
        )
        .agg(
            quantidade_operacoes=(
                "id_operacao",
                "nunique",
            ),

            total_produzido=(
                "total_produzido",
                "sum",
            ),

            total_vendido=(
                "total_vendido",
                "sum",
            ),

            total_sobra=(
                "total_sobra",
                "sum",
            ),

            media_produzida_por_operacao=(
                "total_produzido",
                "mean",
            ),

            media_vendida_por_operacao=(
                "total_vendido",
                "mean",
            ),

            mediana_vendida_por_operacao=(
                "total_vendido",
                "median",
            ),

            media_sobra_por_operacao=(
                "total_sobra",
                "mean",
            ),
        )
        .reset_index()
    )

    result[
        "taxa_sobra_percentual"
    ] = result.apply(
        lambda row: calculate_leftover_rate(
            leftover=row["total_sobra"],
            produced=row["total_produzido"],
        ),
        axis=1,
    )

    numeric_columns = [
        "media_produzida_por_operacao",
        "media_vendida_por_operacao",
        "mediana_vendida_por_operacao",
        "media_sobra_por_operacao",
        "taxa_sobra_percentual",
    ]

    for column in numeric_columns:
        result[column] = (
            result[column].round(2)
        )

    return result.sort_values(
        "media_vendida_por_operacao",
        ascending=False,
    ).reset_index(
        drop=True
    )


def analyze_holidays(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compara feriados e dias normais em nível de operação.

    Isso evita utilizar média por registro de produto,
    que poderia distorcer a comparação.
    """

    operations = create_operation_level_data(
        dataframe
    )

    operations["tipo_dia"] = (
        operations[
            "eh_feriado"
        ].map(
            {
                0: "Dia normal",
                1: "Feriado",
            }
        )
    )

    result = (
        operations.groupby(
            "tipo_dia",
            dropna=False,
        )
        .agg(
            quantidade_operacoes=(
                "id_operacao",
                "nunique",
            ),

            total_produzido=(
                "total_produzido",
                "sum",
            ),

            total_vendido=(
                "total_vendido",
                "sum",
            ),

            total_sobra=(
                "total_sobra",
                "sum",
            ),

            media_produzida_por_operacao=(
                "total_produzido",
                "mean",
            ),

            media_vendida_por_operacao=(
                "total_vendido",
                "mean",
            ),

            mediana_vendida_por_operacao=(
                "total_vendido",
                "median",
            ),

            media_sobra_por_operacao=(
                "total_sobra",
                "mean",
            ),
        )
        .reset_index()
    )

    result[
        "taxa_sobra_percentual"
    ] = result.apply(
        lambda row: calculate_leftover_rate(
            leftover=row["total_sobra"],
            produced=row["total_produzido"],
        ),
        axis=1,
    )

    numeric_columns = [
        "media_produzida_por_operacao",
        "media_vendida_por_operacao",
        "mediana_vendida_por_operacao",
        "media_sobra_por_operacao",
        "taxa_sobra_percentual",
    ]

    for column in numeric_columns:
        result[column] = (
            result[column].round(2)
        )

    return result.reset_index(
        drop=True
    )


def analyze_by_date(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega produção, venda e sobra por data.

    Neste caso, SAB_C + SAB_E e DOM_C + DOM_E são somadas
    quando ocorrem na mesma data.

    Portanto, essa tabela representa a demanda total do
    negócio naquela data.
    """

    df = prepare_analysis_data(
        dataframe
    )

    result = _create_group_analysis(
        df,
        ["data_venda"],
    )

    return result.sort_values(
        "data_venda"
    ).reset_index(
        drop=True
    )


def analyze_by_date_and_fair(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Agrega os dados separadamente por data e feira.

    Essa tabela permite analisar cada feira como uma série
    temporal distinta.
    """

    df = prepare_analysis_data(
        dataframe
    )

    result = _create_group_analysis(
        df,
        [
            "data_venda",
            "feira",
        ],
    )

    return result.sort_values(
        [
            "data_venda",
            "feira",
        ]
    ).reset_index(
        drop=True
    )


def run_exploratory_analysis(
    dataframe: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """
    Executa todas as análises exploratórias.
    """

    analyses = {
        "resumo_geral": (
            create_general_summary(
                dataframe
            )
        ),

        "analise_por_feira": (
            analyze_by_fair(
                dataframe
            )
        ),

        "analise_por_categoria": (
            analyze_by_category(
                dataframe
            )
        ),

        "analise_por_produto": (
            analyze_by_product(
                dataframe
            )
        ),

        "analise_por_clima": (
            analyze_by_climate(
                dataframe
            )
        ),

        "analise_feriados": (
            analyze_holidays(
                dataframe
            )
        ),

        "analise_por_data": (
            analyze_by_date(
                dataframe
            )
        ),

        "analise_por_data_feira": (
            analyze_by_date_and_fair(
                dataframe
            )
        ),
    }

    return analyses


def save_exploratory_tables(
    analyses: dict[str, pd.DataFrame],
) -> list[Path]:
    """
    Salva as tabelas da análise exploratória em CSV.
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

    generated_files: list[Path] = []

    for name, dataframe in analyses.items():
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