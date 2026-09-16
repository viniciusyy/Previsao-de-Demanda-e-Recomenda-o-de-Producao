"""
suavização exponencial simples e regressão linear.

Todos os ajustes são refeitos dentro de cada fold temporal. A suavização
é ajustada separadamente por série, enquanto a regressão utiliza um modelo
global com feira, categoria e clima como atributos categóricos.
"""

from math import isfinite
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from forecasting.baselines import calculate_grouped_metrics


TARGET_COLUMN = "demanda_observada"

NUMERIC_FEATURES = [
    "eh_feriado",
    "tendencia_serie",
    "lag_1",
    "lag_2",
    "lag_3",
]

CATEGORICAL_FEATURES = [
    "feira",
    "categoria",
    "clima",
]

REGRESSION_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

ALPHA_CANDIDATES = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
]

NEW_MODELS = {
    "suavizacao_exponencial_simples": "Suavização exponencial simples",
    "regressao_linear_global": "Regressão linear global",
}


def prepare_model_data(fold_details: pd.DataFrame) -> pd.DataFrame:
    """Valida e ordena os dados usados pelos modelos da Fase 9."""

    if fold_details.empty:
        raise ValueError("O conjunto de folds temporais está vazio.")

    required_columns = [
        "fold",
        "conjunto",
        "id_linha_modelagem",
        "data_venda",
        "serie",
        TARGET_COLUMN,
        *REGRESSION_FEATURES,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in fold_details.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(missing_columns)
        )

    dataframe = fold_details.copy()
    dataframe["data_venda"] = pd.to_datetime(dataframe["data_venda"])
    dataframe[TARGET_COLUMN] = pd.to_numeric(
        dataframe[TARGET_COLUMN], errors="raise"
    )

    return dataframe.sort_values(
        ["fold", "conjunto", "serie", "data_venda"]
    ).reset_index(drop=True)


def _simple_exponential_forecast(
    values: list[float],
    alpha: float,
) -> float:
    """Calcula a previsão seguinte por suavização exponencial simples."""

    level = float(values[0])

    for observed in values[1:]:
        level = alpha * float(observed) + (1 - alpha) * level

    return float(level)


def _select_alpha(values: list[float]) -> tuple[float, float]:
    """Escolhe o alpha com menor erro quadrático somente no treino."""

    if len(values) < 2:
        raise ValueError(
            "São necessárias pelo menos duas observações para selecionar alpha."
        )

    best_alpha = ALPHA_CANDIDATES[0]
    best_mse = float("inf")

    for alpha in ALPHA_CANDIDATES:
        level = float(values[0])
        squared_errors = []

        for observed in values[1:]:
            forecast = level
            error = float(observed) - forecast
            squared_errors.append(error**2)
            level = alpha * float(observed) + (1 - alpha) * level

        mse = sum(squared_errors) / len(squared_errors)

        if mse < best_mse - 1e-12:
            best_alpha = alpha
            best_mse = mse

    return float(best_alpha), float(best_mse)


def create_exponential_smoothing_predictions(
    fold_details: pd.DataFrame,
) -> pd.DataFrame:
    """Gera uma previsão por série e fold com suavização simples."""

    dataframe = prepare_model_data(fold_details)
    rows = []

    for fold in sorted(dataframe["fold"].unique()):
        current = dataframe.loc[dataframe["fold"] == fold]
        train = current.loc[current["conjunto"] == "treino"]
        test = current.loc[current["conjunto"] == "teste"]

        for _, test_row in test.iterrows():
            series_train = train.loc[
                train["serie"] == test_row["serie"]
            ].sort_values("data_venda")

            values = (
                pd.to_numeric(series_train[TARGET_COLUMN], errors="raise")
                .astype(float)
                .tolist()
            )

            alpha, training_mse = _select_alpha(values)
            forecast = _simple_exponential_forecast(values, alpha)

            rows.append(
                {
                    "fold": int(fold),
                    "id_linha_modelagem": int(test_row["id_linha_modelagem"]),
                    "data_venda": test_row["data_venda"],
                    "serie": test_row["serie"],
                    "feira": test_row["feira"],
                    "categoria": test_row["categoria"],
                    "modelo": "suavizacao_exponencial_simples",
                    "nome_modelo": NEW_MODELS[
                        "suavizacao_exponencial_simples"
                    ],
                    "valor_real": float(test_row[TARGET_COLUMN]),
                    "previsao_bruta": forecast,
                    "previsao": forecast,
                    "ajustada_para_zero": 0,
                    "alpha": alpha,
                    "mse_treino_selecao_alpha": training_mse,
                    "linhas_treino_modelo": len(series_train),
                }
            )

    return pd.DataFrame(rows)


def create_linear_regression_pipeline() -> Pipeline:
    """Cria o pipeline ajustado novamente em cada fold."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputacao", SimpleImputer(strategy="median")),
            ("padronizacao", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputacao", SimpleImputer(strategy="most_frequent")),
            (
                "codificacao",
                OneHotEncoder(
                    handle_unknown="ignore",
                    drop="first",
                ),
            ),
        ]
    )

    preprocessing = ColumnTransformer(
        transformers=[
            ("numericos", numeric_pipeline, NUMERIC_FEATURES),
            (
                "categoricos",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )

    return Pipeline(
        steps=[
            ("pre_processamento", preprocessing),
            ("modelo", LinearRegression()),
        ]
    )


def create_linear_regression_predictions(
    fold_details: pd.DataFrame,
) -> pd.DataFrame:
    """Treina uma regressão global em cada fold e gera as previsões."""

    dataframe = prepare_model_data(fold_details)
    rows = []

    for fold in sorted(dataframe["fold"].unique()):
        current = dataframe.loc[dataframe["fold"] == fold]
        train = current.loc[current["conjunto"] == "treino"].copy()
        test = current.loc[current["conjunto"] == "teste"].copy()

        pipeline = create_linear_regression_pipeline()
        pipeline.fit(train[REGRESSION_FEATURES], train[TARGET_COLUMN])
        raw_predictions = pipeline.predict(test[REGRESSION_FEATURES])

        for (_, test_row), raw_prediction in zip(
            test.iterrows(), raw_predictions
        ):
            raw_forecast = float(raw_prediction)
            forecast = max(raw_forecast, 0.0)

            rows.append(
                {
                    "fold": int(fold),
                    "id_linha_modelagem": int(test_row["id_linha_modelagem"]),
                    "data_venda": test_row["data_venda"],
                    "serie": test_row["serie"],
                    "feira": test_row["feira"],
                    "categoria": test_row["categoria"],
                    "modelo": "regressao_linear_global",
                    "nome_modelo": NEW_MODELS["regressao_linear_global"],
                    "valor_real": float(test_row[TARGET_COLUMN]),
                    "previsao_bruta": raw_forecast,
                    "previsao": forecast,
                    "ajustada_para_zero": int(raw_forecast < 0),
                    "alpha": float("nan"),
                    "mse_treino_selecao_alpha": float("nan"),
                    "linhas_treino_modelo": len(train),
                }
            )

    return pd.DataFrame(rows)


def add_error_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta os erros usados nas métricas."""

    result = predictions.copy()
    result["erro"] = result["valor_real"] - result["previsao"]
    result["erro_absoluto"] = result["erro"].abs()
    result["erro_quadratico"] = result["erro"] ** 2
    result["erro_percentual_absoluto"] = (
        result["erro_absoluto"] / result["valor_real"].abs() * 100
    )
    result.loc[
        result["valor_real"].eq(0),
        "erro_percentual_absoluto",
    ] = float("nan")

    return result


def create_model_ranking(
    comparison_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Ordena todos os modelos com MAE como critério principal."""

    ranking = comparison_metrics.sort_values(
        ["mae", "rmse", "mape"],
        ascending=True,
        na_position="last",
    ).reset_index(drop=True)
    ranking.insert(0, "posicao", range(1, len(ranking) + 1))
    ranking["criterio_ordenacao"] = "MAE; desempate por RMSE e MAPE"
    return ranking


def create_phase_decision(
    ranking: pd.DataFrame,
    baseline_decision: pd.DataFrame,
) -> pd.DataFrame:
    """Registra o candidato líder depois da Fase 9."""

    best = ranking.iloc[0]
    baseline = baseline_decision.iloc[0]
    baseline_mae = float(baseline["mae"])
    best_mae = float(best["mae"])
    improvement = (baseline_mae - best_mae) / baseline_mae * 100

    return pd.DataFrame(
        [
            {
                "modelo_lider_fase9": best["modelo"],
                "nome_modelo": best["nome_modelo"],
                "mae": best["mae"],
                "mse": best["mse"],
                "rmse": best["rmse"],
                "mape": best["mape"],
                "baseline_referencia": baseline["modelo_referencia"],
                "mae_baseline_referencia": baseline_mae,
                "superou_baseline_em_mae": bool(best_mae < baseline_mae),
                "melhoria_percentual_mae_sobre_baseline": round(
                    improvement, 4
                ),
                "status": (
                    "Candidato líder após a Fase 9; a MLP ainda não foi avaliada."
                ),
            }
        ]
    )


def create_phase_configuration() -> pd.DataFrame:
    """Documenta as configurações metodológicas da Fase 9."""

    rows = [
        (
            "suavizacao_exponencial",
            "simples, sem tendência e sem sazonalidade",
            "Adequada ao histórico curto e ajustada separadamente por série.",
        ),
        (
            "selecao_alpha",
            ", ".join(str(value) for value in ALPHA_CANDIDATES),
            "O alpha com menor MSE interno é escolhido somente no treino.",
        ),
        (
            "regressao_linear",
            "modelo global ajustado por fold",
            "Compartilha informação entre as 18 séries.",
        ),
        (
            "atributos_numericos",
            ", ".join(NUMERIC_FEATURES),
            "Somente informações conhecidas ou históricas no momento da previsão.",
        ),
        (
            "atributos_categoricos",
            ", ".join(CATEGORICAL_FEATURES),
            "Codificados dentro do treino de cada fold.",
        ),
        (
            "limite_inferior_previsao",
            "zero",
            "Demanda prevista não pode ser negativa.",
        ),
        (
            "ajuste_pre_processamento",
            "somente no treino de cada fold",
            "Evita vazamento de informação do teste.",
        ),
    ]

    return pd.DataFrame(rows, columns=["parametro", "valor", "justificativa"])


def validate_phase_models(
    new_predictions: pd.DataFrame,
    comparison_predictions: pd.DataFrame,
    fold_details: pd.DataFrame,
    comparison_metrics: pd.DataFrame,
) -> pd.DataFrame:
    """Valida cobertura, ordem temporal e resultados dos modelos."""

    checks: list[dict[str, str]] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "teste": name,
                "resultado": "OK" if passed else "FALHA",
                "detalhes": details,
            }
        )

    expected_models = set(NEW_MODELS)
    found_models = set(new_predictions["modelo"].unique())
    add_check(
        "Modelos da Fase 9 executados",
        found_models == expected_models,
        "Modelos encontrados: " + ", ".join(sorted(found_models)),
    )

    test_rows = fold_details.loc[fold_details["conjunto"] == "teste"]
    expected_count = len(test_rows)
    counts = new_predictions.groupby("modelo").size().to_dict()
    add_check(
        "Novos modelos com todos os casos de teste",
        all(value == expected_count for value in counts.values())
        and len(counts) == len(expected_models),
        f"Esperado por modelo: {expected_count}; encontrado: {counts}",
    )

    expected_pairs = set(
        zip(
            test_rows["fold"].astype(int),
            test_rows["id_linha_modelagem"].astype(int),
        )
    )
    same_cases = all(
        set(
            zip(
                group["fold"].astype(int),
                group["id_linha_modelagem"].astype(int),
            )
        )
        == expected_pairs
        for _, group in new_predictions.groupby("modelo")
    )
    add_check(
        "Mesmos casos de teste da validação temporal",
        same_cases,
        f"Casos esperados por modelo: {len(expected_pairs)}",
    )

    missing = int(
        new_predictions[["valor_real", "previsao"]].isna().sum().sum()
    )
    finite = all(
        isfinite(float(value))
        for value in new_predictions[["valor_real", "previsao"]]
        .stack()
        .tolist()
    )
    add_check(
        "Previsões válidas",
        missing == 0 and finite,
        f"Valores ausentes: {missing}",
    )

    negatives = int(new_predictions["previsao"].lt(0).sum())
    clipped = int(new_predictions["ajustada_para_zero"].sum())
    add_check(
        "Previsões finais não negativas",
        negatives == 0,
        f"Negativas finais: {negatives}; regressões ajustadas para zero: {clipped}",
    )

    smoothing = new_predictions.loc[
        new_predictions["modelo"] == "suavizacao_exponencial_simples"
    ]
    valid_alphas = smoothing["alpha"].isin(ALPHA_CANDIDATES).all()
    add_check(
        "Alphas escolhidos somente entre os candidatos",
        valid_alphas,
        "Alphas utilizados: "
        + ", ".join(str(value) for value in sorted(smoothing["alpha"].unique())),
    )

    regression = new_predictions.loc[
        new_predictions["modelo"] == "regressao_linear_global"
    ]
    expected_train_sizes = (
        fold_details.loc[fold_details["conjunto"] == "treino"]
        .groupby("fold")
        .size()
        .to_dict()
    )
    training_sizes_ok = all(
        group["linhas_treino_modelo"].eq(expected_train_sizes[int(fold)]).all()
        for fold, group in regression.groupby("fold")
    )
    add_check(
        "Regressão reajustada em cada fold",
        training_sizes_ok and regression["fold"].nunique() == 5,
        f"Tamanhos esperados de treino: {expected_train_sizes}",
    )

    comparison_counts = comparison_predictions.groupby("modelo").size()
    add_check(
        "Comparação com quantidade uniforme de previsões",
        comparison_counts.eq(expected_count).all(),
        f"Previsões por modelo: {comparison_counts.to_dict()}",
    )

    metric_columns = ["mae", "mse", "rmse"]
    absolute_metrics_valid = all(
        isfinite(float(value))
        for value in comparison_metrics[metric_columns].stack().tolist()
    )
    valid_mape = comparison_metrics.loc[
        comparison_metrics["quantidade_previsoes_mape"].gt(0),
        "mape",
    ]
    mape_valid = all(isfinite(float(value)) for value in valid_mape)
    add_check(
        "Métricas da comparação válidas",
        absolute_metrics_valid and mape_valid,
        "MAE, MSE e RMSE devem ser finitos; MAPE é validado quando há "
        "alvos não nulos.",
    )

    return pd.DataFrame(checks)


def run_statistical_model_evaluation(
    fold_details: pd.DataFrame,
    baseline_analyses: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Executa a Fase 9 e compara os resultados com os baselines."""

    smoothing = create_exponential_smoothing_predictions(fold_details)
    regression = create_linear_regression_predictions(fold_details)
    new_predictions = add_error_columns(
        pd.concat([smoothing, regression], ignore_index=True)
    )

    baseline_predictions = baseline_analyses["previsoes_baselines"].copy()
    comparison_predictions = pd.concat(
        [baseline_predictions, new_predictions],
        ignore_index=True,
        sort=False,
    )

    new_general_metrics = calculate_grouped_metrics(
        new_predictions, ["modelo", "nome_modelo"]
    )
    new_fold_metrics = calculate_grouped_metrics(
        new_predictions, ["modelo", "nome_modelo", "fold"]
    )
    new_series_metrics = calculate_grouped_metrics(
        new_predictions,
        ["modelo", "nome_modelo", "serie", "feira", "categoria"],
    )
    comparison_metrics = calculate_grouped_metrics(
        comparison_predictions, ["modelo", "nome_modelo"]
    )
    comparison_fold_metrics = calculate_grouped_metrics(
        comparison_predictions, ["modelo", "nome_modelo", "fold"]
    )

    validation = validate_phase_models(
        new_predictions,
        comparison_predictions,
        fold_details,
        comparison_metrics,
    )

    if validation["resultado"].eq("FALHA").any():
        failed_rows = validation.loc[
            validation["resultado"] == "FALHA", ["teste", "detalhes"]
        ]
        failed = [
            f"{row['teste']} ({row['detalhes']})"
            for _, row in failed_rows.iterrows()
        ]
        raise ValueError(
            "Falha na avaliação dos modelos da Fase 9: " + "; ".join(failed)
        )

    ranking = create_model_ranking(comparison_metrics)
    decision = create_phase_decision(
        ranking,
        baseline_analyses["decisao_baseline_referencia"],
    )

    alpha_summary = (
        smoothing[
            [
                "fold",
                "serie",
                "feira",
                "categoria",
                "alpha",
                "mse_treino_selecao_alpha",
                "linhas_treino_modelo",
            ]
        ]
        .sort_values(["fold", "serie"])
        .reset_index(drop=True)
    )

    return {
        "previsoes_modelos_fase9": new_predictions,
        "metricas_modelos_fase9_geral": new_general_metrics,
        "metricas_modelos_fase9_por_fold": new_fold_metrics,
        "metricas_modelos_fase9_por_serie": new_series_metrics,
        "metricas_comparacao_fase9_geral": comparison_metrics,
        "metricas_comparacao_fase9_por_fold": comparison_fold_metrics,
        "ranking_modelos_fase9": ranking,
        "validacao_modelos_fase9": validation,
        "configuracao_modelos_fase9": create_phase_configuration(),
        "alfas_suavizacao_fase9": alpha_summary,
        "decisao_modelos_fase9": decision,
        "previsoes_comparacao_fase9": comparison_predictions,
    }


def save_statistical_model_outputs(
    analyses: dict[str, pd.DataFrame],
) -> list[Path]:
    """Salva os artefatos tabulares da Fase 9."""

    project_root = Path(__file__).resolve().parent.parent
    tables_directory = project_root / "reports" / "tables"
    experiments_directory = project_root / "reports" / "experiments"
    tables_directory.mkdir(parents=True, exist_ok=True)
    experiments_directory.mkdir(parents=True, exist_ok=True)

    experiment_names = {
        "previsoes_modelos_fase9",
        "previsoes_comparacao_fase9",
    }
    generated_files = []

    for name, dataframe in analyses.items():
        directory = (
            experiments_directory if name in experiment_names else tables_directory
        )
        path = directory / f"{name}.csv"
        dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        generated_files.append(path)

    return generated_files
