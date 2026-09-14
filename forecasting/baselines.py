"""
Modelos de referência para previsão de demanda.

Os baselines utilizam somente o histórico de treinamento
disponível em cada fold temporal.
"""

from math import isfinite, sqrt
from pathlib import Path

import pandas as pd


TARGET_COLUMN = "demanda_observada"

BASELINE_CONFIGURATIONS = [
    {
        "modelo": "naive",
        "nome_modelo": "Naive (última ocorrência)",
        "janela": 1,
        "atributo_auditoria": "lag_1",
    },
    {
        "modelo": "media_movel_2",
        "nome_modelo": "Média móvel (2 ocorrências)",
        "janela": 2,
        "atributo_auditoria": "media_movel_2",
    },
    {
        "modelo": "media_movel_3",
        "nome_modelo": "Média móvel (3 ocorrências)",
        "janela": 3,
        "atributo_auditoria": "media_movel_3",
    },
    {
        "modelo": "media_movel_4",
        "nome_modelo": "Média móvel (4 ocorrências)",
        "janela": 4,
        "atributo_auditoria": "media_movel_4",
    },
]


def prepare_baseline_data(
    fold_details: pd.DataFrame,
) -> pd.DataFrame:
    """Valida e organiza os dados dos folds."""

    if fold_details.empty:
        raise ValueError(
            "O conjunto de folds temporais está vazio."
        )

    audit_columns = [
        configuration["atributo_auditoria"]
        for configuration in BASELINE_CONFIGURATIONS
    ]

    required_columns = [
        "fold",
        "conjunto",
        "id_linha_modelagem",
        "data_venda",
        "serie",
        "feira",
        "categoria",
        TARGET_COLUMN,
        *audit_columns,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in fold_details.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes: "
            + ", ".join(missing_columns)
        )

    dataframe = fold_details.copy()

    dataframe["data_venda"] = pd.to_datetime(
        dataframe["data_venda"]
    )

    dataframe[TARGET_COLUMN] = pd.to_numeric(
        dataframe[TARGET_COLUMN],
        errors="raise",
    )

    dataframe = dataframe.sort_values(
        [
            "fold",
            "conjunto",
            "serie",
            "data_venda",
        ]
    ).reset_index(drop=True)

    if not {"treino", "teste"}.issubset(
        set(dataframe["conjunto"].unique())
    ):
        raise ValueError(
            "Os folds precisam possuir linhas "
            "de treino e teste."
        )

    return dataframe


def create_baseline_predictions(
    fold_details: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gera previsões usando somente o treino de cada fold.

    A coluna equivalente criada na Fase 6 é mantida
    apenas para auditoria da previsão.
    """

    dataframe = prepare_baseline_data(
        fold_details
    )

    prediction_rows = []

    for fold in sorted(dataframe["fold"].unique()):
        current_fold = dataframe.loc[
            dataframe["fold"] == fold
        ]

        train = current_fold.loc[
            current_fold["conjunto"] == "treino"
        ].copy()

        test = current_fold.loc[
            current_fold["conjunto"] == "teste"
        ].copy()

        if train.empty or test.empty:
            raise ValueError(
                f"O Fold {fold} não possui "
                "treino ou teste."
            )

        for _, test_row in test.iterrows():
            series = test_row["serie"]

            series_history = train.loc[
                train["serie"] == series
            ].sort_values("data_venda")

            if series_history.empty:
                raise ValueError(
                    f"A série {series} não possui "
                    "histórico de treinamento "
                    f"no Fold {fold}."
                )

            history_values = pd.to_numeric(
                series_history[TARGET_COLUMN],
                errors="raise",
            )

            for configuration in (
                BASELINE_CONFIGURATIONS
            ):
                window = int(
                    configuration["janela"]
                )

                if len(history_values) < window:
                    raise ValueError(
                        "Histórico insuficiente para "
                        f"{series}, Fold {fold} e "
                        f"janela {window}."
                    )

                if window == 1:
                    forecast = float(
                        history_values.iloc[-1]
                    )
                else:
                    forecast = float(
                        history_values
                        .tail(window)
                        .mean()
                    )

                audit_column = str(
                    configuration[
                        "atributo_auditoria"
                    ]
                )

                prediction_rows.append(
                    {
                        "fold": int(fold),
                        "id_linha_modelagem": int(
                            test_row[
                                "id_linha_modelagem"
                            ]
                        ),
                        "data_venda": test_row[
                            "data_venda"
                        ],
                        "serie": series,
                        "feira": test_row["feira"],
                        "categoria": test_row[
                            "categoria"
                        ],
                        "modelo": configuration[
                            "modelo"
                        ],
                        "nome_modelo": configuration[
                            "nome_modelo"
                        ],
                        "janela_historica": window,
                        "valor_real": float(
                            test_row[TARGET_COLUMN]
                        ),
                        "previsao": forecast,
                        "atributo_auditoria": (
                            audit_column
                        ),
                        "previsao_atributo_fase_6": (
                            float(
                                test_row[
                                    audit_column
                                ]
                            )
                        ),
                    }
                )

    predictions = pd.DataFrame(
        prediction_rows
    )

    predictions["erro"] = (
        predictions["valor_real"]
        - predictions["previsao"]
    )

    predictions["erro_absoluto"] = (
        predictions["erro"].abs()
    )

    predictions["erro_quadratico"] = (
        predictions["erro"] ** 2
    )

    predictions[
        "erro_percentual_absoluto"
    ] = (
        predictions["erro_absoluto"]
        / predictions["valor_real"].abs()
        * 100
    )

    zero_target = predictions[
        "valor_real"
    ].eq(0)

    predictions.loc[
        zero_target,
        "erro_percentual_absoluto",
    ] = float("nan")

    return predictions.sort_values(
        [
            "modelo",
            "fold",
            "serie",
            "data_venda",
        ]
    ).reset_index(drop=True)


def calculate_metric_row(
    group: pd.DataFrame,
) -> dict[str, float | int]:
    """Calcula as métricas de um grupo de previsões."""

    mse = float(
        group["erro_quadratico"].mean()
    )

    valid_percentage_errors = group[
        "erro_percentual_absoluto"
    ].dropna()

    if valid_percentage_errors.empty:
        mape = float("nan")
    else:
        mape = float(
            valid_percentage_errors.mean()
        )

    return {
        "quantidade_previsoes": len(group),
        "quantidade_alvos_zero": int(
            group["valor_real"].eq(0).sum()
        ),
        "quantidade_previsoes_mape": len(
            valid_percentage_errors
        ),
        "demanda_media_observada": float(
            group["valor_real"].mean()
        ),
        "mae": float(
            group["erro_absoluto"].mean()
        ),
        "mse": mse,
        "rmse": sqrt(mse),
        "mape": mape,
    }


def calculate_grouped_metrics(
    predictions: pd.DataFrame,
    grouping_columns: list[str],
) -> pd.DataFrame:
    """Calcula métricas para as colunas de agrupamento."""

    rows = []

    if len(grouping_columns) == 1:
        grouping_argument = grouping_columns[0]
    else:
        grouping_argument = grouping_columns

    groups = predictions.groupby(
        grouping_argument,
        dropna=False,
        sort=False,
    )

    for group_values, group in groups:
        if len(grouping_columns) == 1:
            values = (group_values,)
        else:
            values = tuple(group_values)

        row = dict(
            zip(
                grouping_columns,
                values,
            )
        )

        row.update(
            calculate_metric_row(group)
        )

        rows.append(row)

    result = pd.DataFrame(rows)

    metric_columns = [
        "demanda_media_observada",
        "mae",
        "mse",
        "rmse",
        "mape",
    ]

    result[metric_columns] = (
        result[metric_columns].round(4)
    )

    return result.sort_values(
        grouping_columns
    ).reset_index(drop=True)


def create_baseline_ranking(
    general_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """
    Ordena os baselines.

    O MAE é o critério principal. RMSE e MAPE são
    usados como critérios de desempate.
    """

    ranking = general_metrics.copy()

    ranking = ranking.sort_values(
        ["mae", "rmse", "mape"],
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)

    ranking.insert(
        0,
        "posicao",
        range(1, len(ranking) + 1),
    )

    ranking["criterio_ordenacao"] = (
        "MAE; desempate por RMSE e MAPE"
    )

    return ranking


def create_baseline_decision(
    ranking: pd.DataFrame,
) -> pd.DataFrame:
    """Registra o melhor baseline como referência."""

    best = ranking.iloc[0]

    return pd.DataFrame(
        [
            {
                "modelo_referencia": best[
                    "modelo"
                ],
                "nome_modelo": best[
                    "nome_modelo"
                ],
                "criterio_principal": (
                    "menor MAE geral"
                ),
                "criterios_desempate": (
                    "menor RMSE e menor MAPE"
                ),
                "mae": best["mae"],
                "mse": best["mse"],
                "rmse": best["rmse"],
                "mape": best["mape"],
                "quantidade_previsoes": best[
                    "quantidade_previsoes"
                ],
                "status": (
                    "Baseline de referência; "
                    "não representa o modelo final."
                ),
            }
        ]
    )


def validate_baseline_predictions(
    predictions: pd.DataFrame,
    fold_details: pd.DataFrame,
    general_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Valida previsões, cobertura e métricas."""

    checks: list[dict[str, str]] = []

    def add_check(
        name: str,
        passed: bool,
        details: str,
    ) -> None:
        checks.append(
            {
                "teste": name,
                "resultado": (
                    "OK" if passed else "FALHA"
                ),
                "detalhes": details,
            }
        )

    expected_models = {
        configuration["modelo"]
        for configuration in BASELINE_CONFIGURATIONS
    }

    found_models = set(
        predictions["modelo"].unique()
    )

    add_check(
        "Todos os baselines foram executados",
        found_models == expected_models,
        (
            "Modelos encontrados: "
            + ", ".join(sorted(found_models))
        ),
    )

    test_rows = fold_details.loc[
        fold_details["conjunto"] == "teste"
    ]

    expected_predictions_per_model = len(
        test_rows
    )

    prediction_counts = (
        predictions.groupby("modelo")
        .size()
        .to_dict()
    )

    correct_quantity = (
        len(prediction_counts)
        == len(expected_models)
        and all(
            quantity
            == expected_predictions_per_model
            for quantity
            in prediction_counts.values()
        )
    )

    add_check(
        "Mesma quantidade de previsões por modelo",
        correct_quantity,
        (
            "Esperado por modelo: "
            f"{expected_predictions_per_model}; "
            f"encontrado: {prediction_counts}"
        ),
    )

    expected_pairs = set(
        zip(
            test_rows["fold"].astype(int),
            test_rows[
                "id_linha_modelagem"
            ].astype(int),
        )
    )

    same_test_rows = True

    for model in expected_models:
        model_predictions = predictions.loc[
            predictions["modelo"] == model
        ]

        model_pairs = set(
            zip(
                model_predictions[
                    "fold"
                ].astype(int),
                model_predictions[
                    "id_linha_modelagem"
                ].astype(int),
            )
        )

        if model_pairs != expected_pairs:
            same_test_rows = False
            break

    add_check(
        "Mesmos casos de teste para todos os modelos",
        same_test_rows,
        (
            "Casos de teste esperados por modelo: "
            f"{len(expected_pairs)}"
        ),
    )

    numeric_columns = [
        "valor_real",
        "previsao",
        "previsao_atributo_fase_6",
    ]

    missing_values = int(
        predictions[numeric_columns]
        .isna()
        .sum()
        .sum()
    )

    finite_values = all(
        isfinite(float(value))
        for value in (
            predictions[numeric_columns]
            .stack()
            .tolist()
        )
    )

    add_check(
        "Previsões sem valores ausentes ou infinitos",
        missing_values == 0 and finite_values,
        (
            f"Valores ausentes: {missing_values}"
        ),
    )

    negative_predictions = int(
        predictions["previsao"].lt(0).sum()
    )

    add_check(
        "Previsões não negativas",
        negative_predictions == 0,
        (
            "Previsões negativas encontradas: "
            f"{negative_predictions}"
        ),
    )

    # As médias móveis da Fase 6 foram armazenadas
    # com duas casas decimais. As previsões permanecem
    # com precisão completa para o cálculo das métricas.
    rounded_predictions = (
        predictions["previsao"].round(2)
    )

    audit_difference = (
        rounded_predictions
        - predictions[
            "previsao_atributo_fase_6"
        ]
    ).abs()

    maximum_difference = float(
        audit_difference.max()
    )

    add_check(
        "Baselines usam somente histórico anterior",
        maximum_difference <= 1e-9,
        (
            "Maior diferença após arredondamento "
            "de auditoria para duas casas: "
            f"{maximum_difference}"
        ),
    )

    zero_target = predictions[
        "valor_real"
    ].eq(0)

    correct_mape_treatment = (
        predictions.loc[
            zero_target,
            "erro_percentual_absoluto",
        ].isna().all()
        and predictions.loc[
            ~zero_target,
            "erro_percentual_absoluto",
        ].notna().all()
    )

    add_check(
        "Tratamento de alvos iguais a zero no MAPE",
        correct_mape_treatment,
        (
            "Alvos iguais a zero: "
            f"{int(zero_target.sum())}"
        ),
    )

    valid_metrics = True

    for _, row in general_metrics.iterrows():
        basic_metrics = [
            row["mae"],
            row["mse"],
            row["rmse"],
        ]

        if not all(
            isfinite(float(value))
            for value in basic_metrics
        ):
            valid_metrics = False
            break

        if (
            row["quantidade_previsoes_mape"] > 0
            and not isfinite(float(row["mape"]))
        ):
            valid_metrics = False
            break

    add_check(
        "Métricas calculadas corretamente",
        valid_metrics,
        (
            "MAE, MSE e RMSE devem ser finitos; "
            "MAPE é calculado somente para "
            "alvos não nulos."
        ),
    )

    return pd.DataFrame(checks)


def run_baseline_evaluation(
    fold_details: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Executa os baselines e calcula suas métricas."""

    predictions = create_baseline_predictions(
        fold_details
    )

    general_metrics = calculate_grouped_metrics(
        predictions,
        ["modelo", "nome_modelo"],
    )

    fold_metrics = calculate_grouped_metrics(
        predictions,
        [
            "modelo",
            "nome_modelo",
            "fold",
        ],
    )

    series_metrics = calculate_grouped_metrics(
        predictions,
        [
            "modelo",
            "nome_modelo",
            "serie",
            "feira",
            "categoria",
        ],
    )

    validation = validate_baseline_predictions(
        predictions=predictions,
        fold_details=fold_details,
        general_metrics=general_metrics,
    )

    if (
        validation["resultado"] == "FALHA"
    ).any():
        failed_rows = validation.loc[
            validation["resultado"] == "FALHA",
            ["teste", "detalhes"],
        ]

        failed = [
            (
                f"{row['teste']} "
                f"({row['detalhes']})"
            )
            for _, row in failed_rows.iterrows()
        ]

        raise ValueError(
            "Falha na avaliação dos baselines: "
            + "; ".join(failed)
        )

    ranking = create_baseline_ranking(
        general_metrics
    )

    decision = create_baseline_decision(
        ranking
    )

    return {
        "previsoes_baselines": predictions,
        "metricas_baselines_geral": (
            general_metrics
        ),
        "metricas_baselines_por_fold": (
            fold_metrics
        ),
        "metricas_baselines_por_serie": (
            series_metrics
        ),
        "ranking_baselines": ranking,
        "validacao_baselines": validation,
        "decisao_baseline_referencia": decision,
    }


def save_baseline_outputs(
    analyses: dict[str, pd.DataFrame],
) -> list[Path]:
    """Salva os artefatos da Fase 8."""

    project_root = (
        Path(__file__).resolve().parent.parent
    )

    tables_directory = (
        project_root / "reports" / "tables"
    )

    experiments_directory = (
        project_root / "reports" / "experiments"
    )

    tables_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    experiments_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    destinations = {
        "previsoes_baselines": (
            experiments_directory
        ),
        "metricas_baselines_geral": (
            tables_directory
        ),
        "metricas_baselines_por_fold": (
            tables_directory
        ),
        "metricas_baselines_por_serie": (
            tables_directory
        ),
        "ranking_baselines": (
            tables_directory
        ),
        "validacao_baselines": (
            tables_directory
        ),
        "decisao_baseline_referencia": (
            tables_directory
        ),
    }

    generated_files = []

    for name, dataframe in analyses.items():
        path = (
            destinations[name]
            / f"{name}.csv"
        )

        dataframe.to_csv(
            path,
            index=False,
            encoding="utf-8-sig",
        )

        generated_files.append(path)

    return generated_files