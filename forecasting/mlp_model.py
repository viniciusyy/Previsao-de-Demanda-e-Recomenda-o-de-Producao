"""
Modelo MLP global .

A configuração da rede é escolhida por uma validação temporal interna
realizada somente no treino de cada fold. Depois da seleção, a rede é
reajustada com todo o treino do fold e avaliada no mesmo caso de teste
utilizado pelos demais modelos.
"""

from math import isfinite, sqrt
from pathlib import Path
import warnings

import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from forecasting.baselines import calculate_grouped_metrics
from forecasting.statistical_models import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    REGRESSION_FEATURES,
    TARGET_COLUMN,
)


MODEL_KEY = "mlp_global"
MODEL_NAME = "MLP global"
RANDOM_STATE = 42
MAX_ITERATIONS = 2000
MAX_FUNCTION_EVALUATIONS = 30000

MLP_CONFIGURATIONS = [
    {
        "configuracao": "mlp_8_alpha_0001",
        "camadas_ocultas": (8,),
        "alpha": 0.001,
    },
    {
        "configuracao": "mlp_8_alpha_001",
        "camadas_ocultas": (8,),
        "alpha": 0.01,
    },
    {
        "configuracao": "mlp_16_alpha_001",
        "camadas_ocultas": (16,),
        "alpha": 0.01,
    },
    {
        "configuracao": "mlp_8_4_alpha_001",
        "camadas_ocultas": (8, 4),
        "alpha": 0.01,
    },
]


def prepare_mlp_data(fold_details: pd.DataFrame) -> pd.DataFrame:
    """Valida e ordena os dados utilizados pela MLP."""

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


def create_mlp_pipeline(configuration: dict) -> TransformedTargetRegressor:
    """Cria o pipeline completo da MLP para um fold."""

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

    regression_pipeline = Pipeline(
        steps=[
            ("pre_processamento", preprocessing),
            (
                "modelo",
                MLPRegressor(
                    hidden_layer_sizes=configuration["camadas_ocultas"],
                    activation="relu",
                    solver="lbfgs",
                    alpha=configuration["alpha"],
                    max_iter=MAX_ITERATIONS,
                    max_fun=MAX_FUNCTION_EVALUATIONS,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    return TransformedTargetRegressor(
        regressor=regression_pipeline,
        transformer=StandardScaler(),
    )


def _fit_mlp(
    train: pd.DataFrame,
    configuration: dict,
) -> tuple[TransformedTargetRegressor, dict[str, object]]:
    """Ajusta uma MLP e registra informações de convergência."""

    model = create_mlp_pipeline(configuration)

    with warnings.catch_warnings(record=True) as captured_warnings:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(
            train[REGRESSION_FEATURES],
            train[TARGET_COLUMN],
        )

    convergence_warning = any(
        issubclass(item.category, ConvergenceWarning)
        for item in captured_warnings
    )

    fitted_mlp = model.regressor_.named_steps["modelo"]
    diagnostics = {
        "iteracoes": int(fitted_mlp.n_iter_),
        "perda_final": float(fitted_mlp.loss_),
        "alerta_convergencia": int(convergence_warning),
        "atingiu_limite_iteracoes": int(
            fitted_mlp.n_iter_ >= MAX_ITERATIONS
        ),
    }

    return model, diagnostics


def create_inner_temporal_split(
    outer_train: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, bool]:
    """Reserva a última observação de cada série para validação interna."""

    ordered = outer_train.sort_values(
        ["serie", "data_venda"]
    ).copy()
    validation_indices = ordered.groupby("serie").tail(1).index

    inner_validation = ordered.loc[validation_indices].copy()
    inner_train = ordered.drop(index=validation_indices).copy()

    temporal_order_ok = True

    for series, validation_group in inner_validation.groupby("serie"):
        series_train = inner_train.loc[inner_train["serie"] == series]

        if series_train.empty:
            temporal_order_ok = False
            break

        if series_train["data_venda"].max() >= validation_group[
            "data_venda"
        ].min():
            temporal_order_ok = False
            break

    return inner_train, inner_validation, temporal_order_ok


def _calculate_prediction_metrics(
    observed: pd.Series,
    predicted: pd.Series,
) -> dict[str, float | int]:
    """Calcula métricas usadas na seleção interna."""

    errors = observed.astype(float) - predicted.astype(float)
    absolute_errors = errors.abs()
    squared_errors = errors**2
    valid_percentage = observed.ne(0)

    mse = float(squared_errors.mean())
    mape = float("nan")

    if valid_percentage.any():
        mape = float(
            (
                absolute_errors.loc[valid_percentage]
                / observed.loc[valid_percentage].abs()
                * 100
            ).mean()
        )

    return {
        "mae_validacao_interna": float(absolute_errors.mean()),
        "mse_validacao_interna": mse,
        "rmse_validacao_interna": sqrt(mse),
        "mape_validacao_interna": mape,
        "alvos_zero_validacao_interna": int(observed.eq(0).sum()),
    }


def _format_hidden_layers(layers: tuple[int, ...]) -> str:
    """Converte a arquitetura para uma representação estável no CSV."""

    return "x".join(str(neurons) for neurons in layers)


def select_mlp_configuration(
    training_data: pd.DataFrame,
) -> tuple[dict, pd.DataFrame]:
    """Seleciona a configuração da MLP usando somente dados históricos.

    Esta função também é utilizada na previsão operacional. A última
    observação disponível de cada série é reservada para a validação interna;
    nenhuma observação futura é usada na escolha da arquitetura.
    """

    required_columns = [
        "data_venda",
        "serie",
        TARGET_COLUMN,
        *REGRESSION_FEATURES,
    ]
    missing_columns = [
        column
        for column in required_columns
        if column not in training_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes na seleção operacional da MLP: "
            + ", ".join(missing_columns)
        )

    ordered = training_data.copy()
    ordered["data_venda"] = pd.to_datetime(
        ordered["data_venda"], errors="raise"
    )
    ordered[TARGET_COLUMN] = pd.to_numeric(
        ordered[TARGET_COLUMN], errors="raise"
    )
    ordered = ordered.sort_values(
        ["serie", "data_venda"]
    ).reset_index(drop=True)

    inner_train, inner_validation, temporal_order_ok = (
        create_inner_temporal_split(ordered)
    )

    if inner_train.empty or inner_validation.empty:
        raise ValueError(
            "Não há histórico suficiente para selecionar a configuração "
            "operacional da MLP."
        )

    selection_rows = []

    for configuration in MLP_CONFIGURATIONS:
        model, diagnostics = _fit_mlp(inner_train, configuration)
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=(
                    "Found unknown categories in columns .* during "
                    "transform.*"
                ),
                category=UserWarning,
            )
            raw_predictions = model.predict(
                inner_validation[REGRESSION_FEATURES]
            )
        predictions = pd.Series(raw_predictions).clip(lower=0)
        observed = inner_validation[TARGET_COLUMN].reset_index(drop=True)
        metrics = _calculate_prediction_metrics(observed, predictions)

        selection_rows.append(
            {
                "configuracao": configuration["configuracao"],
                "camadas_ocultas": _format_hidden_layers(
                    configuration["camadas_ocultas"]
                ),
                "alpha": float(configuration["alpha"]),
                "solver": "lbfgs",
                "funcao_ativacao": "relu",
                "random_state": RANDOM_STATE,
                "linhas_treino_interno": len(inner_train),
                "linhas_validacao_interna": len(inner_validation),
                "series_validacao_interna": inner_validation[
                    "serie"
                ].nunique(),
                "ordem_temporal_interna_ok": temporal_order_ok,
                "previsoes_negativas_brutas_validacao": int(
                    (raw_predictions < 0).sum()
                ),
                **metrics,
                **diagnostics,
            }
        )

    selection = pd.DataFrame(selection_rows).sort_values(
        [
            "mae_validacao_interna",
            "rmse_validacao_interna",
            "configuracao",
        ]
    ).reset_index(drop=True)
    selected_name = str(selection.iloc[0]["configuracao"])
    selected_configuration = next(
        configuration.copy()
        for configuration in MLP_CONFIGURATIONS
        if configuration["configuracao"] == selected_name
    )
    selection["selecionada"] = selection["configuracao"].eq(
        selected_name
    )

    return selected_configuration, selection


def fit_mlp_model(
    training_data: pd.DataFrame,
    configuration: dict,
) -> tuple[TransformedTargetRegressor, dict[str, object]]:
    """Ajusta a MLP final com a configuração previamente selecionada."""

    return _fit_mlp(training_data, configuration)


def create_mlp_predictions(
    fold_details: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Seleciona a configuração por fold e gera as previsões externas."""

    dataframe = prepare_mlp_data(fold_details)
    prediction_rows = []
    selection_rows = []

    for fold in sorted(dataframe["fold"].unique()):
        current = dataframe.loc[dataframe["fold"] == fold]
        outer_train = current.loc[current["conjunto"] == "treino"].copy()
        outer_test = current.loc[current["conjunto"] == "teste"].copy()

        (
            inner_train,
            inner_validation,
            temporal_order_ok,
        ) = create_inner_temporal_split(outer_train)

        fold_selection_rows = []

        for configuration in MLP_CONFIGURATIONS:
            model, diagnostics = _fit_mlp(inner_train, configuration)
            raw_predictions = model.predict(
                inner_validation[REGRESSION_FEATURES]
            )
            predictions = pd.Series(raw_predictions).clip(lower=0)
            observed = inner_validation[TARGET_COLUMN].reset_index(drop=True)
            metrics = _calculate_prediction_metrics(observed, predictions)

            row = {
                "fold": int(fold),
                "configuracao": configuration["configuracao"],
                "camadas_ocultas": _format_hidden_layers(
                    configuration["camadas_ocultas"]
                ),
                "alpha": float(configuration["alpha"]),
                "solver": "lbfgs",
                "funcao_ativacao": "relu",
                "random_state": RANDOM_STATE,
                "linhas_treino_interno": len(inner_train),
                "linhas_validacao_interna": len(inner_validation),
                "series_validacao_interna": inner_validation[
                    "serie"
                ].nunique(),
                "ordem_temporal_interna_ok": temporal_order_ok,
                "previsoes_negativas_brutas_validacao": int(
                    (raw_predictions < 0).sum()
                ),
                **metrics,
                **diagnostics,
            }
            fold_selection_rows.append(row)

        fold_selection = pd.DataFrame(fold_selection_rows).sort_values(
            [
                "mae_validacao_interna",
                "rmse_validacao_interna",
                "configuracao",
            ]
        )
        selected_row = fold_selection.iloc[0]
        selected_configuration = next(
            configuration
            for configuration in MLP_CONFIGURATIONS
            if configuration["configuracao"]
            == selected_row["configuracao"]
        )

        fold_selection["selecionada"] = (
            fold_selection["configuracao"]
            == selected_configuration["configuracao"]
        )
        selection_rows.extend(fold_selection.to_dict("records"))

        final_model, final_diagnostics = _fit_mlp(
            outer_train,
            selected_configuration,
        )
        raw_test_predictions = final_model.predict(
            outer_test[REGRESSION_FEATURES]
        )

        for (_, test_row), raw_prediction in zip(
            outer_test.iterrows(), raw_test_predictions
        ):
            raw_forecast = float(raw_prediction)
            forecast = max(raw_forecast, 0.0)

            prediction_rows.append(
                {
                    "fold": int(fold),
                    "id_linha_modelagem": int(
                        test_row["id_linha_modelagem"]
                    ),
                    "data_venda": test_row["data_venda"],
                    "serie": test_row["serie"],
                    "feira": test_row["feira"],
                    "categoria": test_row["categoria"],
                    "modelo": MODEL_KEY,
                    "nome_modelo": MODEL_NAME,
                    "valor_real": float(test_row[TARGET_COLUMN]),
                    "previsao_bruta": raw_forecast,
                    "previsao": forecast,
                    "ajustada_para_zero": int(raw_forecast < 0),
                    "configuracao_selecionada": selected_configuration[
                        "configuracao"
                    ],
                    "camadas_ocultas": _format_hidden_layers(
                        selected_configuration["camadas_ocultas"]
                    ),
                    "alpha": float(selected_configuration["alpha"]),
                    "solver": "lbfgs",
                    "random_state": RANDOM_STATE,
                    "linhas_treino_modelo": len(outer_train),
                    "mae_validacao_interna": float(
                        selected_row["mae_validacao_interna"]
                    ),
                    "iteracoes_modelo_final": final_diagnostics[
                        "iteracoes"
                    ],
                    "perda_modelo_final": final_diagnostics[
                        "perda_final"
                    ],
                    "alerta_convergencia_modelo_final": final_diagnostics[
                        "alerta_convergencia"
                    ],
                    "atingiu_limite_iteracoes_modelo_final": (
                        final_diagnostics["atingiu_limite_iteracoes"]
                    ),
                }
            )

    predictions = pd.DataFrame(prediction_rows)
    selection = pd.DataFrame(selection_rows).sort_values(
        ["fold", "configuracao"]
    ).reset_index(drop=True)

    return predictions, selection


def add_error_columns(predictions: pd.DataFrame) -> pd.DataFrame:
    """Acrescenta os erros usados na avaliação externa."""

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
    """Ordena os sete modelos com MAE como critério principal."""

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
    comparison_metrics: pd.DataFrame,
    phase9_analyses: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Registra a liderança após incluir a MLP."""

    best = ranking.iloc[0]
    mlp = comparison_metrics.loc[
        comparison_metrics["modelo"] == MODEL_KEY
    ].iloc[0]
    phase9_decision = phase9_analyses["decisao_modelos_fase9"].iloc[0]
    baseline_key = phase9_decision["baseline_referencia"]
    baseline = comparison_metrics.loc[
        comparison_metrics["modelo"] == baseline_key
    ].iloc[0]

    baseline_mae = float(baseline["mae"])
    mlp_mae = float(mlp["mae"])
    improvement = (baseline_mae - mlp_mae) / baseline_mae * 100

    return pd.DataFrame(
        [
            {
                "modelo_lider_fase10": best["modelo"],
                "nome_modelo_lider": best["nome_modelo"],
                "mae_lider": best["mae"],
                "mse_lider": best["mse"],
                "rmse_lider": best["rmse"],
                "mape_lider": best["mape"],
                "baseline_referencia": baseline_key,
                "nome_baseline_referencia": baseline["nome_modelo"],
                "mae_baseline_referencia": baseline_mae,
                "mae_mlp": mlp_mae,
                "rmse_mlp": mlp["rmse"],
                "mape_mlp": mlp["mape"],
                "mlp_superou_baseline_em_mae": bool(
                    mlp_mae < baseline_mae
                ),
                "melhoria_percentual_mae_mlp_sobre_baseline": round(
                    improvement, 4
                ),
                "status": (
                    "Liderança após comparar a MLP com os seis modelos "
                    "anteriores."
                ),
            }
        ]
    )


def create_phase_configuration() -> pd.DataFrame:
    """Documenta as decisões metodológicas da MLP."""

    configurations = "; ".join(
        (
            f"{item['configuracao']}="
            f"{_format_hidden_layers(item['camadas_ocultas'])}, "
            f"alpha={item['alpha']}"
        )
        for item in MLP_CONFIGURATIONS
    )

    rows = [
        (
            "tipo_modelo",
            "MLP global",
            "Compartilha informação entre as 18 séries.",
        ),
        (
            "configuracoes_candidatas",
            configurations,
            "Busca pequena para limitar o risco de sobreajuste.",
        ),
        (
            "selecao_configuracao",
            "validação temporal interna por fold",
            "A última observação de treino de cada série é usada somente "
            "para selecionar a configuração.",
        ),
        (
            "solver",
            "lbfgs",
            "Adequado a conjuntos de dados de menor dimensão.",
        ),
        (
            "ativacao",
            "relu",
            "Introduz relações não lineares entre os atributos.",
        ),
        (
            "random_state",
            str(RANDOM_STATE),
            "Mantém a execução reprodutível.",
        ),
        (
            "max_iter",
            str(MAX_ITERATIONS),
            "Limite de otimização acompanhado por diagnóstico.",
        ),
        (
            "atributos_numericos",
            ", ".join(NUMERIC_FEATURES),
            "Mesmos atributos numéricos da regressão linear.",
        ),
        (
            "atributos_categoricos",
            ", ".join(CATEGORICAL_FEATURES),
            "Mesmos atributos categóricos da regressão linear.",
        ),
        (
            "padronizacao_alvo",
            "StandardScaler por fold",
            "Facilita a otimização da rede e é ajustado somente no treino.",
        ),
        (
            "limite_inferior_previsao",
            "zero",
            "Demanda prevista não pode ser negativa.",
        ),
    ]

    return pd.DataFrame(rows, columns=["parametro", "valor", "justificativa"])


def validate_mlp_phase(
    mlp_predictions: pd.DataFrame,
    selection: pd.DataFrame,
    comparison_predictions: pd.DataFrame,
    comparison_metrics: pd.DataFrame,
    fold_details: pd.DataFrame,
) -> pd.DataFrame:
    """Audita cobertura, seleção interna e métricas da Fase 10."""

    checks: list[dict[str, str]] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "teste": name,
                "resultado": "OK" if passed else "FALHA",
                "detalhes": details,
            }
        )

    test_rows = fold_details.loc[fold_details["conjunto"] == "teste"]
    expected_count = len(test_rows)
    add_check(
        "MLP executada em todos os casos de teste",
        len(mlp_predictions) == expected_count,
        f"Esperado: {expected_count}; encontrado: {len(mlp_predictions)}",
    )

    expected_pairs = set(
        zip(
            test_rows["fold"].astype(int),
            test_rows["id_linha_modelagem"].astype(int),
        )
    )
    found_pairs = set(
        zip(
            mlp_predictions["fold"].astype(int),
            mlp_predictions["id_linha_modelagem"].astype(int),
        )
    )
    add_check(
        "MLP usa os mesmos casos da validação temporal",
        found_pairs == expected_pairs,
        f"Casos esperados: {len(expected_pairs)}; encontrados: "
        f"{len(found_pairs)}",
    )

    missing = int(
        mlp_predictions[["valor_real", "previsao"]].isna().sum().sum()
    )
    finite = all(
        isfinite(float(value))
        for value in mlp_predictions[["valor_real", "previsao"]]
        .stack()
        .tolist()
    )
    add_check(
        "Previsões da MLP válidas",
        missing == 0 and finite,
        f"Valores ausentes: {missing}",
    )

    negatives = int(mlp_predictions["previsao"].lt(0).sum())
    clipped = int(mlp_predictions["ajustada_para_zero"].sum())
    add_check(
        "Previsões finais da MLP não negativas",
        negatives == 0,
        f"Negativas finais: {negatives}; ajustadas para zero: {clipped}",
    )

    expected_train_sizes = (
        fold_details.loc[fold_details["conjunto"] == "treino"]
        .groupby("fold")
        .size()
        .to_dict()
    )
    training_sizes_ok = all(
        group["linhas_treino_modelo"].eq(
            expected_train_sizes[int(fold)]
        ).all()
        for fold, group in mlp_predictions.groupby("fold")
    )
    add_check(
        "MLP reajustada com todo o treino de cada fold",
        training_sizes_ok,
        f"Tamanhos esperados: {expected_train_sizes}",
    )

    inner_order_ok = selection["ordem_temporal_interna_ok"].all()
    inner_series_ok = selection["series_validacao_interna"].eq(18).all()
    add_check(
        "Seleção interna respeita a ordem temporal",
        bool(inner_order_ok and inner_series_ok),
        "A validação interna deve possuir a última observação de cada uma "
        "das 18 séries.",
    )

    expected_configurations = {
        item["configuracao"] for item in MLP_CONFIGURATIONS
    }
    configurations_ok = all(
        set(group["configuracao"]) == expected_configurations
        and int(group["selecionada"].sum()) == 1
        for _, group in selection.groupby("fold")
    )
    add_check(
        "Todas as configurações avaliadas dentro de cada fold",
        configurations_ok,
        f"Configurações esperadas: {sorted(expected_configurations)}",
    )

    seeds_ok = (
        mlp_predictions["random_state"].eq(RANDOM_STATE).all()
        and selection["random_state"].eq(RANDOM_STATE).all()
    )
    add_check(
        "Execução reprodutível",
        bool(seeds_ok),
        f"random_state esperado: {RANDOM_STATE}",
    )

    prohibited_features = {
        "demanda_observada",
        "total_produzido",
        "total_sobra",
        "quantidade_produzida",
        "quantidade_sobra",
    }
    leaked_features = sorted(
        prohibited_features.intersection(REGRESSION_FEATURES)
    )
    add_check(
        "Atributos sem informação posterior à previsão",
        len(leaked_features) == 0,
        "Atributos proibidos encontrados: "
        + (", ".join(leaked_features) if leaked_features else "nenhum"),
    )

    convergence_warnings = int(
        mlp_predictions[
            "alerta_convergencia_modelo_final"
        ].groupby(mlp_predictions["fold"]).max().sum()
    )
    iterations_limit = int(
        mlp_predictions[
            "atingiu_limite_iteracoes_modelo_final"
        ].groupby(mlp_predictions["fold"]).max().sum()
    )
    add_check(
        "Convergência dos modelos finais monitorada",
        convergence_warnings == 0 and iterations_limit == 0,
        f"Folds com alerta: {convergence_warnings}; folds no limite de "
        f"iterações: {iterations_limit}",
    )

    comparison_counts = comparison_predictions.groupby("modelo").size()
    add_check(
        "Comparação com quantidade uniforme de previsões",
        comparison_counts.eq(expected_count).all(),
        f"Previsões por modelo: {comparison_counts.to_dict()}",
    )

    absolute_metric_columns = ["mae", "mse", "rmse"]
    absolute_metrics_valid = all(
        isfinite(float(value))
        for value in comparison_metrics[absolute_metric_columns]
        .stack()
        .tolist()
    )
    valid_mape = comparison_metrics.loc[
        comparison_metrics["quantidade_previsoes_mape"].gt(0),
        "mape",
    ]
    mape_valid = all(isfinite(float(value)) for value in valid_mape)
    add_check(
        "Métricas da Fase 10 válidas",
        absolute_metrics_valid and mape_valid,
        "MAE, MSE e RMSE devem ser finitos; MAPE é validado quando há "
        "alvos não nulos.",
    )

    return pd.DataFrame(checks)


def run_mlp_evaluation(
    fold_details: pd.DataFrame,
    phase9_analyses: dict[str, pd.DataFrame],
) -> dict[str, pd.DataFrame]:
    """Executa a MLP e compara os sete modelos."""

    raw_predictions, selection = create_mlp_predictions(fold_details)
    mlp_predictions = add_error_columns(raw_predictions)

    previous_predictions = phase9_analyses[
        "previsoes_comparacao_fase9"
    ].copy()
    comparison_predictions = pd.concat(
        [previous_predictions, mlp_predictions],
        ignore_index=True,
        sort=False,
    )

    mlp_general_metrics = calculate_grouped_metrics(
        mlp_predictions, ["modelo", "nome_modelo"]
    )
    mlp_fold_metrics = calculate_grouped_metrics(
        mlp_predictions, ["modelo", "nome_modelo", "fold"]
    )
    mlp_series_metrics = calculate_grouped_metrics(
        mlp_predictions,
        ["modelo", "nome_modelo", "serie", "feira", "categoria"],
    )
    comparison_metrics = calculate_grouped_metrics(
        comparison_predictions, ["modelo", "nome_modelo"]
    )
    comparison_fold_metrics = calculate_grouped_metrics(
        comparison_predictions, ["modelo", "nome_modelo", "fold"]
    )

    validation = validate_mlp_phase(
        mlp_predictions,
        selection,
        comparison_predictions,
        comparison_metrics,
        fold_details,
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
            "Falha na avaliação da MLP: " + "; ".join(failed)
        )

    ranking = create_model_ranking(comparison_metrics)
    decision = create_phase_decision(
        ranking,
        comparison_metrics,
        phase9_analyses,
    )

    return {
        "previsoes_mlp": mlp_predictions,
        "metricas_mlp_geral": mlp_general_metrics,
        "metricas_mlp_por_fold": mlp_fold_metrics,
        "metricas_mlp_por_serie": mlp_series_metrics,
        "metricas_comparacao_fase10_geral": comparison_metrics,
        "metricas_comparacao_fase10_por_fold": comparison_fold_metrics,
        "ranking_modelos_fase10": ranking,
        "validacao_mlp": validation,
        "configuracao_mlp": create_phase_configuration(),
        "selecao_configuracao_mlp_por_fold": selection,
        "decisao_modelo_fase10": decision,
        "previsoes_comparacao_fase10": comparison_predictions,
    }


def save_mlp_outputs(
    analyses: dict[str, pd.DataFrame],
) -> list[Path]:
    """Salva os artefatos tabulares."""

    project_root = Path(__file__).resolve().parent.parent
    tables_directory = project_root / "reports" / "tables"
    experiments_directory = project_root / "reports" / "experiments"
    tables_directory.mkdir(parents=True, exist_ok=True)
    experiments_directory.mkdir(parents=True, exist_ok=True)

    experiment_names = {
        "previsoes_mlp",
        "previsoes_comparacao_fase10",
    }
    generated_files = []

    for name, dataframe in analyses.items():
        directory = (
            experiments_directory
            if name in experiment_names
            else tables_directory
        )
        path = directory / f"{name}.csv"
        dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        generated_files.append(path)

    return generated_files
