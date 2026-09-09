"""
Definição da validação temporal do projeto.

A estratégia utilizada é walk-forward com janela expansiva.
Cada fold testa uma ocorrência futura por série e utiliza apenas
observações anteriores daquela série no conjunto de treinamento.
"""

from pathlib import Path

import pandas as pd


NUMBER_OF_FOLDS = 5
MINIMUM_INITIAL_TRAIN = 6
EXPECTED_SERIES = 18


def prepare_temporal_data(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """Ordena o dataset e cria a posição temporal de modelagem."""

    if dataframe.empty:
        raise ValueError(
            "O dataset de modelagem está vazio."
        )

    required_columns = [
        "data_venda",
        "serie",
        "feira",
        "categoria",
        "demanda_observada",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            + ", ".join(missing_columns)
        )

    df = dataframe.copy()
    df["data_venda"] = pd.to_datetime(df["data_venda"])

    df = df.sort_values(
        ["serie", "data_venda"]
    ).reset_index(drop=True)

    df["ordem_modelagem_serie"] = (
        df.groupby("serie").cumcount() + 1
    )
    df["id_linha_modelagem"] = range(1, len(df) + 1)

    counts = df.groupby("serie").size()

    if counts.nunique() != 1:
        raise ValueError(
            "As séries possuem quantidades diferentes de observações. "
            "A configuração temporal deve ser revisada."
        )

    series_count = df["serie"].nunique()

    if series_count != EXPECTED_SERIES:
        raise ValueError(
            f"Eram esperadas {EXPECTED_SERIES} séries, "
            f"mas foram encontradas {series_count}."
        )

    observations_per_series = int(counts.iloc[0])
    initial_train = observations_per_series - NUMBER_OF_FOLDS

    if initial_train < MINIMUM_INITIAL_TRAIN:
        raise ValueError(
            "Histórico insuficiente para criar cinco folds com "
            f"pelo menos {MINIMUM_INITIAL_TRAIN} observações "
            "iniciais de treinamento por série."
        )

    return df


def create_walk_forward_folds(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cria os folds temporais e o resumo de cada rodada."""

    df = prepare_temporal_data(dataframe)

    observations_per_series = int(
        df.groupby("serie").size().iloc[0]
    )

    initial_train = (
        observations_per_series - NUMBER_OF_FOLDS
    )

    fold_frames = []
    summary_rows = []

    for fold in range(1, NUMBER_OF_FOLDS + 1):
        test_position = initial_train + fold

        train = df.loc[
            df["ordem_modelagem_serie"] < test_position
        ].copy()

        test = df.loc[
            df["ordem_modelagem_serie"] == test_position
        ].copy()

        train["fold"] = fold
        train["conjunto"] = "treino"

        test["fold"] = fold
        test["conjunto"] = "teste"

        fold_frames.extend([train, test])

        summary_rows.append(
            {
                "fold": fold,
                "observacoes_treino_por_serie": (
                    test_position - 1
                ),
                "posicao_teste_por_serie": test_position,
                "linhas_treino": len(train),
                "linhas_teste": len(test),
                "series_treino": train["serie"].nunique(),
                "series_teste": test["serie"].nunique(),
                "primeira_data_treino": (
                    train["data_venda"].min()
                ),
                "ultima_data_treino": (
                    train["data_venda"].max()
                ),
                "primeira_data_teste": (
                    test["data_venda"].min()
                ),
                "ultima_data_teste": (
                    test["data_venda"].max()
                ),
                "datas_distintas_teste": (
                    test["data_venda"].nunique()
                ),
            }
        )

    fold_details = pd.concat(
        fold_frames,
        ignore_index=True,
    )

    fold_details = fold_details.sort_values(
        ["fold", "conjunto", "serie", "data_venda"]
    ).reset_index(drop=True)

    fold_summary = pd.DataFrame(summary_rows)

    return fold_details, fold_summary


def validate_temporal_folds(
    fold_details: pd.DataFrame,
    fold_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Valida separação, cobertura e ordem cronológica dos folds."""

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

    for fold in range(1, NUMBER_OF_FOLDS + 1):
        current = fold_details.loc[
            fold_details["fold"] == fold
        ]

        train = current.loc[
            current["conjunto"] == "treino"
        ]

        test = current.loc[
            current["conjunto"] == "teste"
        ]

        overlap = set(
            train["id_linha_modelagem"]
        ).intersection(
            set(test["id_linha_modelagem"])
        )

        add_check(
            f"Fold {fold} sem sobreposição",
            len(overlap) == 0,
            (
                "Linhas presentes nos dois conjuntos: "
                f"{len(overlap)}"
            ),
        )

        one_test_per_series = (
            test.groupby("serie").size().eq(1).all()
            and test["serie"].nunique() == EXPECTED_SERIES
        )

        add_check(
            f"Fold {fold} com uma previsão por série",
            one_test_per_series,
            (
                f"Linhas de teste: {len(test)}; "
                f"séries: {test['serie'].nunique()}"
            ),
        )

        chronological = True

        for series in test["serie"].unique():
            series_train = train.loc[
                train["serie"] == series
            ]

            series_test = test.loc[
                test["serie"] == series
            ]

            if (
                series_train.empty
                or series_test.empty
                or series_train["data_venda"].max()
                >= series_test["data_venda"].min()
            ):
                chronological = False
                break

        add_check(
            f"Fold {fold} respeita a ordem temporal",
            chronological,
            (
                "A última data de treino deve ser anterior "
                "à data de teste em cada série."
            ),
        )

    train_sizes = (
        fold_summary["linhas_treino"].tolist()
    )

    growing_train = all(
        later > earlier
        for earlier, later in zip(
            train_sizes,
            train_sizes[1:],
        )
    )

    add_check(
        "Janela de treinamento expansiva",
        growing_train,
        f"Tamanhos de treino: {train_sizes}",
    )

    total_test_predictions = int(
        fold_summary["linhas_teste"].sum()
    )

    expected_predictions = (
        NUMBER_OF_FOLDS * EXPECTED_SERIES
    )

    add_check(
        "Quantidade total de previsões de teste",
        total_test_predictions == expected_predictions,
        (
            f"Total: {total_test_predictions}; "
            f"esperado: {expected_predictions}"
        ),
    )

    return pd.DataFrame(checks)


def create_temporal_configuration(
    fold_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Documenta as decisões metodológicas da validação."""

    first_fold = fold_summary.iloc[0]
    last_fold = fold_summary.iloc[-1]

    rows = [
        (
            "estrategia",
            "walk-forward com janela expansiva",
            (
                "Preserva a ordem temporal e simula "
                "previsões sucessivas."
            ),
        ),
        (
            "quantidade_folds",
            str(NUMBER_OF_FOLDS),
            (
                "Permite avaliar cinco ocorrências futuras "
                "de cada série."
            ),
        ),
        (
            "series_por_fold",
            str(EXPECTED_SERIES),
            (
                "Cada fold testa todas as combinações "
                "categoria + feira."
            ),
        ),
        (
            "treino_inicial_por_serie",
            str(
                int(
                    first_fold[
                        "observacoes_treino_por_serie"
                    ]
                )
            ),
            (
                "Mantém seis observações iniciais "
                "utilizáveis por série."
            ),
        ),
        (
            "treino_final_por_serie",
            str(
                int(
                    last_fold[
                        "observacoes_treino_por_serie"
                    ]
                )
            ),
            (
                "A janela cresce após cada ocorrência "
                "observada."
            ),
        ),
        (
            "teste_por_fold_e_serie",
            "1",
            (
                "Avaliação de horizonte de uma "
                "ocorrência à frente."
            ),
        ),
        (
            "total_previsoes_teste",
            str(
                int(
                    fold_summary["linhas_teste"].sum()
                )
            ),
            (
                "Base comum para calcular as métricas "
                "de todos os modelos."
            ),
        ),
        (
            "embaralhamento",
            "não",
            (
                "Divisão aleatória causaria vazamento "
                "de informações futuras."
            ),
        ),
        (
            "ajuste_pre_processamento",
            "somente no treino de cada fold",
            (
                "Codificação, normalização e parâmetros "
                "não podem aprender com o teste."
            ),
        ),
        (
            "instantes_de_teste",
            "iguais para todos os modelos",
            (
                "Garante comparação justa entre "
                "os métodos."
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


def run_temporal_validation(
    modeling_data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Executa a definição e a auditoria da validação temporal."""

    (
        fold_details,
        fold_summary,
    ) = create_walk_forward_folds(modeling_data)

    validation = validate_temporal_folds(
        fold_details,
        fold_summary,
    )

    if (validation["resultado"] == "FALHA").any():
        failed = validation.loc[
            validation["resultado"] == "FALHA",
            "teste",
        ].tolist()

        raise ValueError(
            "Falha na validação temporal: "
            + ", ".join(failed)
        )

    configuration = create_temporal_configuration(
        fold_summary
    )

    return {
        "configuracao_validacao_temporal": configuration,
        "resumo_folds_temporais": fold_summary,
        "validacao_folds_temporais": validation,
        "folds_validacao_temporal": fold_details,
    }


def save_temporal_validation_outputs(
    analyses: dict[str, pd.DataFrame],
) -> list[Path]:
    """Salva os artefatos."""

    project_root = Path(__file__).resolve().parent.parent

    processed_directory = (
        project_root / "data" / "processed"
    )

    tables_directory = (
        project_root / "reports" / "tables"
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
        "configuracao_validacao_temporal": (
            tables_directory
        ),
        "resumo_folds_temporais": (
            tables_directory
        ),
        "validacao_folds_temporais": (
            tables_directory
        ),
        "folds_validacao_temporal": (
            processed_directory
        ),
    }

    generated_files = []

    for name, dataframe in analyses.items():
        path = destinations[name] / f"{name}.csv"

        dataframe.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        generated_files.append(path)

    return generated_files