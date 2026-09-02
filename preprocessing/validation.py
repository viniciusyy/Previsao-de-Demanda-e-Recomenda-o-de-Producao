"""
Validação dos dados históricos da pastelaria.

Este módulo identifica inconsistências antes que os registros
sejam utilizados em análise exploratória ou modelos de previsão.

Seu objetivo é apenas:

1. detectar problemas;
2. registrar os problemas;
3. gerar um relatório de validação.
"""

from pathlib import Path

import pandas as pd


# ============================================================
# VALORES VÁLIDOS DA OPERAÇÃO
# ============================================================

VALID_FAIRS = {
    "QUA",
    "QUI",
    "SAB_C",
    "SAB_E",
    "DOM_C",
    "DOM_E",
}


VALID_CLIMATES = {
    "SOL",
    "FRIO",
    "GAROA",
    "CHUVA_MODERADA",
    "CHUVA_FORTE",
}


VALID_CATEGORIES = {
    "Comum",
    "Especial",
    "Doce",
}


VALID_PRODUCT_IDS = set(range(1, 28))


# ============================================================
# PRODUTOS QUE NÃO SÃO COMERCIALIZADOS EM ALGUMAS FEIRAS
# ============================================================

PROHIBITED_PRODUCTS_BY_FAIR = {
    ("SAB_E", 5),
    ("SAB_E", 15),
    ("DOM_E", 5),
    ("DOM_E", 15),
}


# ============================================================
# COLUNAS OBRIGATÓRIAS
# ============================================================

REQUIRED_COLUMNS = [
    "id_operacao",
    "feira",
    "data_producao",
    "data_venda",
    "id_produto",
    "produto",
    "categoria",
    "quantidade_produzida",
    "quantidade_sobra",
    "quantidade_vendida",
    "clima",
    "eh_feriado",
]


# ============================================================
# DIA DA SEMANA ESPERADO PARA CADA FEIRA
# ============================================================
#
# pandas:
#
# segunda = 0
# terça   = 1
# quarta  = 2
# quinta  = 3
# sexta   = 4
# sábado  = 5
# domingo = 6
# ============================================================

EXPECTED_SALE_WEEKDAY = {
    "QUA": 2,
    "QUI": 3,
    "SAB_C": 5,
    "SAB_E": 5,
    "DOM_C": 6,
    "DOM_E": 6,
}


EXPECTED_PRODUCTION_WEEKDAY = {
    "QUA": 1,
    "QUI": 2,
    "SAB_C": 4,
    "SAB_E": 4,
    "DOM_C": 5,
    "DOM_E": 5,
}


def _create_issue_rows(
    dataframe: pd.DataFrame,
    mask: pd.Series,
    validation: str,
    level: str,
    description: str,
) -> list[dict]:
    """
    Cria linhas detalhadas para registros que falharam em uma validação.

    Args:
        dataframe: DataFrame analisado.
        mask: Série booleana indicando registros problemáticos.
        validation: Código da validação.
        level: ERRO, ALERTA ou INFO.
        description: Explicação do problema.

    Returns:
        list[dict]: Lista contendo os problemas encontrados.
    """

    issues = []

    affected_rows = dataframe.loc[mask]

    for index, row in affected_rows.iterrows():
        issues.append(
            {
                "indice_dataframe": index,
                "nivel": level,
                "validacao": validation,
                "id_operacao": row.get("id_operacao"),
                "data_venda": row.get("data_venda"),
                "feira": row.get("feira"),
                "id_produto": row.get("id_produto"),
                "produto": row.get("produto"),
                "descricao": description,
            }
        )

    return issues


def _register_validation(
    summary: list[dict],
    validation: str,
    level: str,
    count: int,
    description: str,
) -> None:
    """
    Registra o resultado resumido de uma validação.
    """

    status = "OK" if count == 0 else "ENCONTRADO"

    summary.append(
        {
            "validacao": validation,
            "nivel": level,
            "quantidade": count,
            "status": status,
            "descricao": description,
        }
    )


def validate_historical_data(
    dataframe: pd.DataFrame,
    referential_issues: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Executa as validações dos dados históricos.

    Args:
        dataframe:
            DataFrame carregado do MySQL.

        referential_issues:
            Problemas encontrados diretamente no banco por meio
            da consulta de integridade referencial.

    Returns:
        tuple:
            Primeiro DataFrame:
                resumo das validações.

            Segundo DataFrame:
                detalhes dos registros problemáticos.
    """

    if dataframe.empty:
        raise ValueError(
            "Não é possível validar uma base histórica vazia."
        )

    df = dataframe.copy()

    summary: list[dict] = []
    detailed_issues: list[dict] = []

    # ========================================================
    # 1. COLUNAS OBRIGATÓRIAS
    # ========================================================

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    _register_validation(
        summary,
        "COLUNAS_OBRIGATORIAS",
        "ERRO",
        len(missing_columns),
        (
            "Verifica se todas as colunas necessárias "
            "estão disponíveis."
        ),
    )

    if missing_columns:
        for column in missing_columns:
            detailed_issues.append(
                {
                    "indice_dataframe": None,
                    "nivel": "ERRO",
                    "validacao": "COLUNA_AUSENTE",
                    "id_operacao": None,
                    "data_venda": None,
                    "feira": None,
                    "id_produto": None,
                    "produto": None,
                    "descricao": (
                        f"Coluna obrigatória ausente: {column}"
                    ),
                }
            )

        return (
            pd.DataFrame(summary),
            pd.DataFrame(detailed_issues),
        )

    # ========================================================
    # 2. VALORES AUSENTES EM CAMPOS OBRIGATÓRIOS
    # ========================================================

    for column in REQUIRED_COLUMNS:
        mask = df[column].isna()

        count = int(mask.sum())

        _register_validation(
            summary,
            f"VALOR_AUSENTE_{column.upper()}",
            "ERRO",
            count,
            f"Verifica valores ausentes na coluna {column}.",
        )

        detailed_issues.extend(
            _create_issue_rows(
                df,
                mask,
                f"VALOR_AUSENTE_{column.upper()}",
                "ERRO",
                f"Valor obrigatório ausente em {column}.",
            )
        )

    # ========================================================
    # PREPARAÇÃO DE CÓPIAS NUMÉRICAS
    # ========================================================

    numeric_columns = [
        "quantidade_produzida",
        "quantidade_sobra",
        "quantidade_vendida",
    ]

    numeric_data: dict[str, pd.Series] = {}

    for column in numeric_columns:
        numeric_data[column] = pd.to_numeric(
            df[column],
            errors="coerce",
        )

        invalid_numeric = (
            df[column].notna()
            & numeric_data[column].isna()
        )

        count = int(invalid_numeric.sum())

        _register_validation(
            summary,
            f"VALOR_NUMERICO_INVALIDO_{column.upper()}",
            "ERRO",
            count,
            f"Verifica se {column} possui valor numérico válido.",
        )

        detailed_issues.extend(
            _create_issue_rows(
                df,
                invalid_numeric,
                f"VALOR_NUMERICO_INVALIDO_{column.upper()}",
                "ERRO",
                f"Valor numérico inválido em {column}.",
            )
        )

    produced = numeric_data["quantidade_produzida"]
    leftover = numeric_data["quantidade_sobra"]
    sold = numeric_data["quantidade_vendida"]

    # ========================================================
    # 3. QUANTIDADE PRODUZIDA NEGATIVA
    # ========================================================

    mask = produced < 0
    count = int(mask.sum())

    _register_validation(
        summary,
        "PRODUCAO_NEGATIVA",
        "ERRO",
        count,
        "Quantidade produzida não pode ser negativa.",
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "PRODUCAO_NEGATIVA",
            "ERRO",
            "Quantidade produzida menor que zero.",
        )
    )

    # ========================================================
    # 4. SOBRA NEGATIVA
    # ========================================================

    mask = leftover < 0
    count = int(mask.sum())

    _register_validation(
        summary,
        "SOBRA_NEGATIVA",
        "ERRO",
        count,
        "Quantidade de sobra não pode ser negativa.",
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "SOBRA_NEGATIVA",
            "ERRO",
            "Quantidade de sobra menor que zero.",
        )
    )

    # ========================================================
    # 5. VENDA NEGATIVA
    # ========================================================

    mask = sold < 0
    count = int(mask.sum())

    _register_validation(
        summary,
        "VENDA_NEGATIVA",
        "ERRO",
        count,
        "Quantidade vendida não pode ser negativa.",
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "VENDA_NEGATIVA",
            "ERRO",
            "Quantidade vendida menor que zero.",
        )
    )

    # ========================================================
    # 6. SOBRA MAIOR QUE PRODUÇÃO
    # ========================================================

    mask = leftover > produced
    count = int(mask.sum())

    _register_validation(
        summary,
        "SOBRA_MAIOR_QUE_PRODUCAO",
        "ERRO",
        count,
        "A sobra não pode ser superior à produção.",
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "SOBRA_MAIOR_QUE_PRODUCAO",
            "ERRO",
            "Quantidade de sobra maior que quantidade produzida.",
        )
    )

    # ========================================================
    # 7. VENDA MAIOR QUE PRODUÇÃO
    # ========================================================

    mask = sold > produced
    count = int(mask.sum())

    _register_validation(
        summary,
        "VENDA_MAIOR_QUE_PRODUCAO",
        "ERRO",
        count,
        "A venda não pode ser superior à produção registrada.",
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "VENDA_MAIOR_QUE_PRODUCAO",
            "ERRO",
            "Quantidade vendida maior que quantidade produzida.",
        )
    )

    # ========================================================
    # 8. RELAÇÃO PRODUÇÃO - SOBRA = VENDIDO
    # ========================================================

    valid_numbers = (
        produced.notna()
        & leftover.notna()
        & sold.notna()
    )

    mask = (
        valid_numbers
        & ((produced - leftover) != sold)
    )

    count = int(mask.sum())

    _register_validation(
        summary,
        "VENDA_INCONSISTENTE",
        "ERRO",
        count,
        (
            "Verifica se vendido = produzido - sobra."
        ),
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "VENDA_INCONSISTENTE",
            "ERRO",
            (
                "Quantidade vendida diferente de "
                "quantidade produzida - quantidade de sobra."
            ),
        )
    )

    # ========================================================
    # 9. DATAS
    # ========================================================

    production_date = pd.to_datetime(
        df["data_producao"],
        errors="coerce",
    )

    sale_date = pd.to_datetime(
        df["data_venda"],
        errors="coerce",
    )

    invalid_production_date = (
        df["data_producao"].notna()
        & production_date.isna()
    )

    count = int(invalid_production_date.sum())

    _register_validation(
        summary,
        "DATA_PRODUCAO_INVALIDA",
        "ERRO",
        count,
        "Verifica se a data de produção é válida.",
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            invalid_production_date,
            "DATA_PRODUCAO_INVALIDA",
            "ERRO",
            "Data de produção inválida.",
        )
    )

    invalid_sale_date = (
        df["data_venda"].notna()
        & sale_date.isna()
    )

    count = int(invalid_sale_date.sum())

    _register_validation(
        summary,
        "DATA_VENDA_INVALIDA",
        "ERRO",
        count,
        "Verifica se a data de venda é válida.",
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            invalid_sale_date,
            "DATA_VENDA_INVALIDA",
            "ERRO",
            "Data de venda inválida.",
        )
    )

    # ========================================================
    # 10. VENDA ANTERIOR À PRODUÇÃO
    # ========================================================

    mask = sale_date < production_date
    count = int(mask.sum())

    _register_validation(
        summary,
        "VENDA_ANTES_DA_PRODUCAO",
        "ERRO",
        count,
        "A venda não pode ocorrer antes da produção.",
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "VENDA_ANTES_DA_PRODUCAO",
            "ERRO",
            "Data de venda anterior à data de produção.",
        )
    )

    # ========================================================
    # 11. INTERVALO ENTRE PRODUÇÃO E VENDA
    # ========================================================
    #
    # Atualmente todas as operações possuem produção
    # no dia anterior à venda.
    #
    # Como podem existir exceções operacionais no futuro,
    # tratamos como ALERTA e não como erro fatal.
    # ========================================================

    interval_days = (
        sale_date - production_date
    ).dt.days

    mask = (
        interval_days.notna()
        & (interval_days != 1)
    )

    count = int(mask.sum())

    _register_validation(
        summary,
        "INTERVALO_PRODUCAO_VENDA",
        "ALERTA",
        count,
        (
            "Atualmente espera-se um dia entre produção "
            "e venda."
        ),
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "INTERVALO_PRODUCAO_VENDA",
            "ALERTA",
            (
                "Intervalo entre produção e venda diferente "
                "de um dia."
            ),
        )
    )

    # ========================================================
    # 12. FEIRA INVÁLIDA
    # ========================================================

    mask = ~df["feira"].isin(VALID_FAIRS)
    count = int(mask.sum())

    _register_validation(
        summary,
        "FEIRA_INVALIDA",
        "ERRO",
        count,
        "Verifica se a feira pertence ao conjunto esperado.",
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "FEIRA_INVALIDA",
            "ERRO",
            "Código de feira inválido.",
        )
    )

    # ========================================================
    # 13. DIA DA SEMANA DA VENDA
    # ========================================================

    expected_sale_day = df["feira"].map(
        EXPECTED_SALE_WEEKDAY
    )

    mask = (
        expected_sale_day.notna()
        & sale_date.notna()
        & (sale_date.dt.weekday != expected_sale_day)
    )

    count = int(mask.sum())

    _register_validation(
        summary,
        "DIA_VENDA_INESPERADO",
        "ALERTA",
        count,
        (
            "Verifica se a data de venda corresponde ao "
            "dia normalmente associado à feira."
        ),
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "DIA_VENDA_INESPERADO",
            "ALERTA",
            "Dia da semana da venda diferente do esperado.",
        )
    )

    # ========================================================
    # 14. DIA DA SEMANA DA PRODUÇÃO
    # ========================================================

    expected_production_day = df["feira"].map(
        EXPECTED_PRODUCTION_WEEKDAY
    )

    mask = (
        expected_production_day.notna()
        & production_date.notna()
        & (
            production_date.dt.weekday
            != expected_production_day
        )
    )

    count = int(mask.sum())

    _register_validation(
        summary,
        "DIA_PRODUCAO_INESPERADO",
        "ALERTA",
        count,
        (
            "Verifica se a produção ocorreu no dia "
            "normalmente associado à feira."
        ),
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "DIA_PRODUCAO_INESPERADO",
            "ALERTA",
            "Dia da semana da produção diferente do esperado.",
        )
    )

    # ========================================================
    # 15. PRODUTO INVÁLIDO
    # ========================================================

    product_ids = pd.to_numeric(
        df["id_produto"],
        errors="coerce",
    )

    mask = ~product_ids.isin(VALID_PRODUCT_IDS)

    count = int(mask.sum())

    _register_validation(
        summary,
        "PRODUTO_INVALIDO",
        "ERRO",
        count,
        "Verifica se o produto pertence aos IDs cadastrados.",
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "PRODUTO_INVALIDO",
            "ERRO",
            "Produto não pertence ao conjunto esperado.",
        )
    )

    # ========================================================
    # 16. CATEGORIA INVÁLIDA
    # ========================================================

    mask = ~df["categoria"].isin(VALID_CATEGORIES)

    count = int(mask.sum())

    _register_validation(
        summary,
        "CATEGORIA_INVALIDA",
        "ERRO",
        count,
        "Verifica a categoria do produto.",
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "CATEGORIA_INVALIDA",
            "ERRO",
            "Categoria inválida.",
        )
    )

    # ========================================================
    # 17. CLIMA INVÁLIDO
    # ========================================================

    mask = ~df["clima"].isin(VALID_CLIMATES)

    count = int(mask.sum())

    _register_validation(
        summary,
        "CLIMA_INVALIDO",
        "ERRO",
        count,
        (
            "Aceita somente SOL, FRIO, GAROA, "
            "CHUVA_MODERADA e CHUVA_FORTE."
        ),
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "CLIMA_INVALIDO",
            "ERRO",
            "Categoria de clima inválida.",
        )
    )

    # ========================================================
    # 18. REGISTROS DUPLICADOS
    # ========================================================
    #
    # Uma operação não deve possuir o mesmo produto
    # registrado mais de uma vez.
    # ========================================================

    mask = df.duplicated(
        subset=[
            "id_operacao",
            "id_produto",
        ],
        keep=False,
    )

    count = int(mask.sum())

    _register_validation(
        summary,
        "REGISTRO_DUPLICADO",
        "ERRO",
        count,
        (
            "Verifica duplicidade da combinação "
            "id_operacao + id_produto."
        ),
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "REGISTRO_DUPLICADO",
            "ERRO",
            (
                "Produto aparece mais de uma vez "
                "na mesma operação."
            ),
        )
    )

    # ========================================================
    # 19. PRODUTOS PROIBIDOS POR FEIRA
    # ========================================================

    prohibited_mask = pd.Series(
        False,
        index=df.index,
    )

    for fair, product_id in PROHIBITED_PRODUCTS_BY_FAIR:
        prohibited_mask |= (
            (df["feira"] == fair)
            & (product_ids == product_id)
        )

    count = int(prohibited_mask.sum())

    _register_validation(
        summary,
        "PRODUTO_PROIBIDO_NA_FEIRA",
        "ERRO",
        count,
        (
            "Frango e Escarola sem bacon não são "
            "comercializados em SAB_E e DOM_E."
        ),
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            prohibited_mask,
            "PRODUTO_PROIBIDO_NA_FEIRA",
            "ERRO",
            "Produto não comercializado nesta feira.",
        )
    )

    # ========================================================
    # 20. INDICADOR DE FERIADO
    # ========================================================

    holiday_indicator = pd.to_numeric(
        df["eh_feriado"],
        errors="coerce",
    )

    mask = ~holiday_indicator.isin([0, 1])

    count = int(mask.sum())

    _register_validation(
        summary,
        "INDICADOR_FERIADO_INVALIDO",
        "ERRO",
        count,
        "eh_feriado deve possuir valor 0 ou 1.",
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "INDICADOR_FERIADO_INVALIDO",
            "ERRO",
            "Indicador de feriado diferente de 0 ou 1.",
        )
    )

    # ========================================================
    # 21. FERIADO SEM NOME
    # ========================================================

    holiday_name_missing = (
        df["nome_feriado"].isna()
        | (
            df["nome_feriado"]
            .astype(str)
            .str.strip()
            .isin(["", "nan", "None"])
        )
    )

    mask = (
        (holiday_indicator == 1)
        & holiday_name_missing
    )

    count = int(mask.sum())

    _register_validation(
        summary,
        "FERIADO_SEM_NOME",
        "ALERTA",
        count,
        (
            "Quando eh_feriado = 1, espera-se que o nome "
            "do feriado esteja informado."
        ),
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "FERIADO_SEM_NOME",
            "ALERTA",
            "Feriado marcado sem nome registrado.",
        )
    )

    # ========================================================
    # 22. DIA NORMAL COM NOME DE FERIADO
    # ========================================================

    holiday_name_present = ~holiday_name_missing

    mask = (
        (holiday_indicator == 0)
        & holiday_name_present
    )

    count = int(mask.sum())

    _register_validation(
        summary,
        "DIA_NORMAL_COM_NOME_FERIADO",
        "ALERTA",
        count,
        (
            "Dia marcado como não feriado possui "
            "nome de feriado preenchido."
        ),
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "DIA_NORMAL_COM_NOME_FERIADO",
            "ALERTA",
            (
                "Nome de feriado preenchido em operação "
                "marcada como dia normal."
            ),
        )
    )

    # ========================================================
    # 23. POSSÍVEL DEMANDA CENSURADA
    # ========================================================
    #
    # NÃO é considerado erro.
    #
    # Quando:
    #
    # sobra = 0
    # vendido = produzido
    #
    # não sabemos se existia demanda adicional.
    # ========================================================

    mask = (
        (produced > 0)
        & (leftover == 0)
        & (sold == produced)
    )

    count = int(mask.sum())

    _register_validation(
        summary,
        "POSSIVEL_DEMANDA_CENSURADA",
        "INFO",
        count,
        (
            "Produto vendeu toda a produção e terminou "
            "sem sobra. Pode ter existido demanda adicional."
        ),
    )

    detailed_issues.extend(
        _create_issue_rows(
            df,
            mask,
            "POSSIVEL_DEMANDA_CENSURADA",
            "INFO",
            (
                "Toda a produção foi vendida. "
                "Não é possível observar eventual demanda perdida."
            ),
        )
    )

    # ========================================================
    # 24. INTEGRIDADE REFERENCIAL DO BANCO
    # ========================================================

    if referential_issues is not None:
        count = len(referential_issues)

        _register_validation(
            summary,
            "INTEGRIDADE_REFERENCIAL",
            "ERRO",
            count,
            (
                "Verifica referências inexistentes entre "
                "as tabelas do banco."
            ),
        )

        if not referential_issues.empty:
            for _, row in referential_issues.iterrows():
                detailed_issues.append(
                    {
                        "indice_dataframe": None,
                        "nivel": "ERRO",
                        "validacao": row.get("tipo"),
                        "id_operacao": row.get("id_operacao"),
                        "data_venda": None,
                        "feira": None,
                        "id_produto": row.get("id_produto"),
                        "produto": None,
                        "descricao": row.get("descricao"),
                    }
                )

    # ========================================================
    # DATAFRAMES FINAIS
    # ========================================================

    summary_df = pd.DataFrame(summary)

    details_columns = [
        "indice_dataframe",
        "nivel",
        "validacao",
        "id_operacao",
        "data_venda",
        "feira",
        "id_produto",
        "produto",
        "descricao",
    ]

    details_df = pd.DataFrame(
        detailed_issues,
        columns=details_columns,
    )

    return summary_df, details_df


def save_validation_reports(
    summary: pd.DataFrame,
    details: pd.DataFrame,
) -> tuple[Path, Path]:
    """
    Salva os relatórios da validação em CSV.

    Returns:
        tuple[Path, Path]:
            Caminho do relatório resumido e detalhado.
    """

    project_root = Path(__file__).resolve().parent.parent

    output_directory = (
        project_root
        / "reports"
        / "tables"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary_path = (
        output_directory
        / "validacao_resumo.csv"
    )

    details_path = (
        output_directory
        / "validacao_detalhes.csv"
    )

    summary.to_csv(
        summary_path,
        index=False,
        encoding="utf-8-sig",
    )

    details.to_csv(
        details_path,
        index=False,
        encoding="utf-8-sig",
    )

    return summary_path, details_path