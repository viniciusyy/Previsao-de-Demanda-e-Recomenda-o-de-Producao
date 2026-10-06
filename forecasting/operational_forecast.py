"""
Previsão operacional.

O modelo selecionado nas fases de avaliação é reaplicado a todo o histórico
disponível para gerar uma previsão de demanda para a próxima ocorrência de
cada combinação categoria + feira. Os sete modelos comparados possuem rotina
operacional. Quando regressão ou MLP vencem, clima e feriado das próximas
feiras são solicitados explicitamente no terminal. Esta fase gera previsões,
não recomendações de produção e não distribui quantidades entre produtos.
"""

from decimal import Decimal, ROUND_HALF_UP
from math import isfinite
from pathlib import Path
from typing import Callable
import warnings

import pandas as pd

from forecasting.mlp_model import (
    fit_mlp_model,
    select_mlp_configuration,
)
from forecasting.statistical_models import (
    REGRESSION_FEATURES,
    TARGET_COLUMN,
    create_linear_regression_pipeline,
    select_exponential_alpha,
    simple_exponential_forecast,
)
from preprocessing.features import HISTORICAL_FEATURES


EXPECTED_SERIES = 18

SUPPORTED_OPERATIONAL_MODELS = {
    "naive": {
        "nome": "Naive (última ocorrência)",
        "janela": 1,
        "estrategia": "media_historica",
        "requer_contexto_futuro": False,
    },
    "media_movel_2": {
        "nome": "Média móvel (2 ocorrências)",
        "janela": 2,
        "estrategia": "media_historica",
        "requer_contexto_futuro": False,
    },
    "media_movel_3": {
        "nome": "Média móvel (3 ocorrências)",
        "janela": 3,
        "estrategia": "media_historica",
        "requer_contexto_futuro": False,
    },
    "media_movel_4": {
        "nome": "Média móvel (4 ocorrências)",
        "janela": 4,
        "estrategia": "media_historica",
        "requer_contexto_futuro": False,
    },
    "suavizacao_exponencial_simples": {
        "nome": "Suavização exponencial simples",
        "janela": None,
        "estrategia": "suavizacao_exponencial",
        "requer_contexto_futuro": False,
    },
    "regressao_linear_global": {
        "nome": "Regressão linear global",
        "janela": 4,
        "estrategia": "regressao_global",
        "requer_contexto_futuro": True,
    },
    "mlp_global": {
        "nome": "MLP global",
        "janela": 4,
        "estrategia": "mlp_global",
        "requer_contexto_futuro": True,
    },
}

CLIMATE_OPTIONS = [
    "Sol",
    "Frio",
    "Garoa",
    "Chuva Moderado",
    "Chuva Forte",
]

CLIMATE_ALIASES = {
    "sol": "Sol",
    "frio": "Frio",
    "garoa": "Garoa",
    "chuva moderado": "Chuva Moderado",
    "chuva moderada": "Chuva Moderado",
    "chuva forte": "Chuva Forte",
}

FAIR_WEEKDAYS = {
    "QUA": 2,
    "QUI": 3,
    "SAB_C": 5,
    "SAB_E": 5,
    "DOM_C": 6,
    "DOM_E": 6,
}

FAIR_ORDER = ["QUA", "QUI", "SAB_C", "SAB_E", "DOM_C", "DOM_E"]


def _round_half_up(value: float) -> int:
    """Arredonda uma demanda para o inteiro mais próximo sem regra bancária."""

    return int(
        Decimal(str(value)).quantize(
            Decimal("1"),
            rounding=ROUND_HALF_UP,
        )
    )


def _next_weekday_after(
    cutoff_date: pd.Timestamp,
    target_weekday: int,
) -> pd.Timestamp:
    """Retorna a primeira ocorrência do dia da semana após a data de corte."""

    days_ahead = (target_weekday - cutoff_date.weekday()) % 7

    if days_ahead == 0:
        days_ahead = 7

    return cutoff_date.normalize() + pd.Timedelta(days=days_ahead)


def get_operational_model(
    phase10_analyses: dict[str, pd.DataFrame],
) -> tuple[str, str, int | None, str, bool]:
    """Obtém o modelo vencedor e sua estratégia operacional."""

    decision = phase10_analyses["decisao_modelo_fase10"].iloc[0]
    model_key = str(decision["modelo_lider_fase10"])

    if model_key not in SUPPORTED_OPERATIONAL_MODELS:
        raise ValueError(
            "O ranking foi alterado com os dados atuais. "
            f"O modelo líder agora é {model_key}, mas ele ainda não possui "
            "uma rotina de previsão operacional implementada. A execução "
            "foi interrompida para revisão metodológica."
        )

    configuration = SUPPORTED_OPERATIONAL_MODELS[model_key]
    model_name = str(
        decision.get("nome_modelo_lider", configuration["nome"])
    )
    configured_window = configuration["janela"]
    history_window = (
        int(configured_window)
        if configured_window is not None
        else None
    )
    strategy = str(configuration["estrategia"])
    requires_future_context = bool(
        configuration["requer_contexto_futuro"]
    )

    return (
        model_key,
        model_name,
        history_window,
        strategy,
        requires_future_context,
    )


def prepare_operational_history(
    category_fair_data: pd.DataFrame,
    history_window: int | None,
    cutoff_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    """Valida e limita o histórico que estará disponível na previsão."""

    if category_fair_data.empty:
        raise ValueError("A base categoria + feira está vazia.")

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
        if column not in category_fair_data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Colunas obrigatórias ausentes: " + ", ".join(missing_columns)
        )

    dataframe = category_fair_data.copy()
    dataframe["data_venda"] = pd.to_datetime(
        dataframe["data_venda"], errors="raise"
    ).dt.normalize()
    dataframe["demanda_observada"] = pd.to_numeric(
        dataframe["demanda_observada"], errors="raise"
    )

    if dataframe["demanda_observada"].lt(0).any():
        raise ValueError("Foram encontradas demandas históricas negativas.")

    duplicated = dataframe.duplicated(
        subset=["data_venda", "feira", "categoria"]
    )

    if duplicated.any():
        raise ValueError(
            "Existem duplicidades na chave data + feira + categoria."
        )

    unknown_fairs = sorted(set(dataframe["feira"]) - set(FAIR_WEEKDAYS))

    if unknown_fairs:
        raise ValueError(
            "Feiras sem calendário configurado: " + ", ".join(unknown_fairs)
        )

    if cutoff_date is None:
        effective_cutoff = dataframe["data_venda"].max()
    else:
        effective_cutoff = pd.Timestamp(cutoff_date).normalize()

    dataframe = dataframe.loc[
        dataframe["data_venda"] <= effective_cutoff
    ].copy()

    if dataframe.empty:
        raise ValueError(
            "Não existem observações até a data de corte informada."
        )

    counts = dataframe.groupby("serie").size()
    minimum_history = history_window if history_window is not None else 2
    insufficient = counts.loc[counts < minimum_history]

    if not insufficient.empty:
        raise ValueError(
            "Histórico insuficiente para o modelo operacional com janela "
            f"mínima de {minimum_history} ocorrência(s): "
            + ", ".join(insufficient.index.astype(str))
        )

    series_count = dataframe["serie"].nunique()

    if series_count != EXPECTED_SERIES:
        raise ValueError(
            f"Eram esperadas {EXPECTED_SERIES} séries, "
            f"mas foram encontradas {series_count}."
        )

    dataframe = dataframe.sort_values(
        ["serie", "data_venda"]
    ).reset_index(drop=True)

    return dataframe, effective_cutoff


def _normalize_climate(value: object) -> str:
    """Normaliza uma das cinco categorias de clima aceitas pelo projeto."""

    normalized = " ".join(str(value).strip().lower().split())

    if normalized not in CLIMATE_ALIASES:
        raise ValueError(
            "Clima inválido. Utilize uma das opções: "
            + ", ".join(CLIMATE_OPTIONS)
            + "."
        )

    return CLIMATE_ALIASES[normalized]


def _normalize_holiday_indicator(value: object) -> int:
    """Converte respostas usuais de feriado para zero ou um."""

    if isinstance(value, bool):
        return int(value)

    normalized = str(value).strip().lower()
    positive = {"1", "s", "sim", "y", "yes", "true"}
    negative = {"0", "n", "nao", "não", "no", "false"}

    if normalized in positive:
        return 1
    if normalized in negative:
        return 0

    raise ValueError("O indicador de feriado deve ser informado como sim ou não.")


def create_future_schedule(
    history: pd.DataFrame,
    cutoff_date: pd.Timestamp,
) -> pd.DataFrame:
    """Cria as próximas datas de produção e venda das seis feiras."""

    fairs = set(history["feira"].astype(str))
    if fairs != set(FAIR_WEEKDAYS):
        raise ValueError(
            "Não foi possível criar o calendário futuro para as seis feiras."
        )

    rows = []
    for fair in FAIR_ORDER:
        sale_date = _next_weekday_after(
            cutoff_date,
            FAIR_WEEKDAYS[fair],
        )
        rows.append(
            {
                "feira": fair,
                "data_producao_prevista": (
                    sale_date - pd.Timedelta(days=1)
                ),
                "data_venda_prevista": sale_date,
            }
        )

    return pd.DataFrame(rows)


def collect_future_context(
    schedule: pd.DataFrame,
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], None] = print,
) -> pd.DataFrame:
    """Solicita clima e feriado para cada feira futura no terminal."""

    output_function("")
    output_function("=" * 70)
    output_function("CONTEXTO DAS PRÓXIMAS FEIRAS")
    output_function("=" * 70)
    output_function("")
    output_function(
        "A MLP/regressão necessita de clima e feriado conhecidos antes "
        "da previsão. Nenhum valor será estimado silenciosamente."
    )

    rows: list[dict[str, object]] = []

    for row in schedule.itertuples(index=False):
        output_function("")
        output_function(
            f"Feira {row.feira} — venda em "
            f"{row.data_venda_prevista:%d/%m/%Y}"
        )
        output_function(
            "Climas aceitos: " + ", ".join(CLIMATE_OPTIONS)
        )

        while True:
            try:
                climate = _normalize_climate(
                    input_function("Clima esperado: ")
                )
                break
            except ValueError as exc:
                output_function(str(exc))

        while True:
            try:
                is_holiday = _normalize_holiday_indicator(
                    input_function("É feriado? [s/n]: ")
                )
                break
            except ValueError as exc:
                output_function(str(exc))

        holiday_name = ""
        if is_holiday:
            while not holiday_name:
                holiday_name = input_function(
                    "Nome do feriado: "
                ).strip()
                if not holiday_name:
                    output_function(
                        "O nome do feriado deve ser informado."
                    )

        rows.append(
            {
                "feira": row.feira,
                "data_producao_prevista": row.data_producao_prevista,
                "data_venda_prevista": row.data_venda_prevista,
                "clima": climate,
                "eh_feriado": is_holiday,
                "nome_feriado": holiday_name,
                "fonte_contexto_futuro": "informado_no_terminal",
            }
        )

    return pd.DataFrame(rows)


def prepare_future_context(
    schedule: pd.DataFrame,
    future_context: pd.DataFrame,
) -> pd.DataFrame:
    """Valida o contexto futuro informado pelo terminal ou por testes."""

    required_columns = ["feira", "clima", "eh_feriado"]
    missing_columns = [
        column
        for column in required_columns
        if column not in future_context.columns
    ]
    if missing_columns:
        raise ValueError(
            "Colunas ausentes no contexto futuro: "
            + ", ".join(missing_columns)
        )

    context = future_context.copy()
    context["feira"] = context["feira"].astype(str)

    if len(context) != len(FAIR_ORDER) or context["feira"].duplicated().any():
        raise ValueError(
            "O contexto futuro deve possuir uma linha para cada uma das "
            "seis feiras."
        )

    if set(context["feira"]) != set(FAIR_ORDER):
        raise ValueError(
            "O contexto futuro não contém exatamente as seis feiras."
        )

    context["clima"] = context["clima"].map(_normalize_climate)
    context["eh_feriado"] = context["eh_feriado"].map(
        _normalize_holiday_indicator
    )

    if "nome_feriado" not in context.columns:
        context["nome_feriado"] = ""
    context["nome_feriado"] = context["nome_feriado"].fillna("").astype(str)

    missing_names = context.loc[
        context["eh_feriado"].eq(1)
        & context["nome_feriado"].str.strip().eq("")
    ]
    if not missing_names.empty:
        raise ValueError(
            "O nome do feriado deve ser informado quando eh_feriado = 1."
        )

    context.loc[
        context["eh_feriado"].eq(0), "nome_feriado"
    ] = ""

    if "fonte_contexto_futuro" not in context.columns:
        context["fonte_contexto_futuro"] = "informado_externamente"

    supplied_date_columns = [
        column
        for column in [
            "data_producao_prevista",
            "data_venda_prevista",
        ]
        if column in context.columns
    ]
    for column in supplied_date_columns:
        context[column] = pd.to_datetime(
            context[column], errors="raise"
        ).dt.normalize()

    context_values = context[
        [
            "feira",
            "clima",
            "eh_feriado",
            "nome_feriado",
            "fonte_contexto_futuro",
            *supplied_date_columns,
        ]
    ]
    merged = schedule.merge(
        context_values,
        on="feira",
        how="left",
        validate="one_to_one",
        suffixes=("", "_informada"),
    )

    for column in supplied_date_columns:
        informed = f"{column}_informada"
        if informed in merged.columns:
            if not merged[column].eq(merged[informed]).all():
                raise ValueError(
                    f"As datas informadas em {column} não correspondem "
                    "ao calendário calculado."
                )
            merged = merged.drop(columns=informed)

    return merged


def create_future_feature_rows(
    history: pd.DataFrame,
    schedule: pd.DataFrame,
    future_context: pd.DataFrame,
) -> pd.DataFrame:
    """Constrói as 18 linhas futuras sem usar demanda posterior ao corte."""

    context = prepare_future_context(schedule, future_context)
    rows: list[dict[str, object]] = []

    for series, group in history.groupby("serie", sort=True):
        ordered = group.sort_values("data_venda")
        recent = ordered.tail(4)

        if len(recent) < 4:
            raise ValueError(
                f"A série {series} não possui quatro observações anteriores."
            )

        fair = str(ordered["feira"].iloc[-1])
        category = str(ordered["categoria"].iloc[-1])
        current_context = context.loc[context["feira"] == fair].iloc[0]
        values = recent["demanda_observada"].astype(float).tolist()

        rows.append(
            {
                "serie": series,
                "feira": fair,
                "categoria": category,
                "data_producao_prevista": current_context[
                    "data_producao_prevista"
                ],
                "data_venda_prevista": current_context[
                    "data_venda_prevista"
                ],
                "clima": current_context["clima"],
                "eh_feriado": int(current_context["eh_feriado"]),
                "nome_feriado": current_context["nome_feriado"],
                "fonte_contexto_futuro": current_context[
                    "fonte_contexto_futuro"
                ],
                "tendencia_serie": int(len(ordered) + 1),
                "lag_1": values[-1],
                "lag_2": values[-2],
                "lag_3": values[-3],
                "media_movel_2": sum(values[-2:]) / 2,
                "media_movel_3": sum(values[-3:]) / 3,
                "media_movel_4": sum(values[-4:]) / 4,
                "quantidade_observacoes_historicas": len(ordered),
            }
        )

    return pd.DataFrame(rows)


def create_history_audit(
    history: pd.DataFrame,
    history_window: int | None,
) -> pd.DataFrame:
    """Documenta as observações históricas utilizadas pelo modelo."""

    rows: list[dict[str, object]] = []

    for series, group in history.groupby("serie", sort=True):
        ordered = group.sort_values("data_venda")
        used = (
            ordered.tail(history_window)
            if history_window is not None
            else ordered
        )
        total = len(used)

        for position, (_, historical_row) in enumerate(
            used.iterrows(), start=1
        ):
            rows.append(
                {
                    "serie": series,
                    "feira": historical_row["feira"],
                    "categoria": historical_row["categoria"],
                    "posicao_na_janela": position,
                    "defasagem": total - position + 1,
                    "data_venda_historica": historical_row["data_venda"],
                    "demanda_observada": float(
                        historical_row["demanda_observada"]
                    ),
                    "utilizada_na_previsao": 1,
                }
            )

    return pd.DataFrame(rows)


def prepare_machine_learning_training_data(
    history: pd.DataFrame,
) -> pd.DataFrame:
    """Recria o mesmo dataset usado na avaliação dos modelos globais."""

    required_columns = [
        "data_venda",
        "serie",
        TARGET_COLUMN,
        *HISTORICAL_FEATURES,
        *REGRESSION_FEATURES,
    ]
    missing_columns = [
        column for column in required_columns if column not in history.columns
    ]
    if missing_columns:
        raise ValueError(
            "Atributos ausentes para o modelo operacional global: "
            + ", ".join(missing_columns)
        )

    training = history.dropna(
        subset=HISTORICAL_FEATURES
    ).copy()

    if training.empty or training[REGRESSION_FEATURES].isna().any().any():
        raise ValueError(
            "O dataset operacional dos modelos globais contém atributos "
            "históricos ausentes."
        )

    return training.sort_values(
        ["serie", "data_venda"]
    ).reset_index(drop=True)


def create_operational_predictions(
    category_fair_data: pd.DataFrame,
    model_key: str,
    model_name: str,
    history_window: int | None,
    strategy: str,
    requires_future_context: bool,
    cutoff_date: str | pd.Timestamp | None = None,
    future_context: pd.DataFrame | None = None,
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], None] = print,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Timestamp,
    dict[str, pd.DataFrame],
]:
    """Gera uma previsão futura com a estratégia do modelo vencedor."""

    dataframe, effective_cutoff = prepare_operational_history(
        category_fair_data,
        history_window,
        cutoff_date,
    )
    schedule = create_future_schedule(dataframe, effective_cutoff)

    prepared_context = pd.DataFrame(
        columns=[
            "feira",
            "data_producao_prevista",
            "data_venda_prevista",
            "clima",
            "eh_feriado",
            "nome_feriado",
            "fonte_contexto_futuro",
        ]
    )
    future_features = pd.DataFrame()

    if requires_future_context:
        if future_context is None:
            future_context = collect_future_context(
                schedule,
                input_function=input_function,
                output_function=output_function,
            )
        prepared_context = prepare_future_context(
            schedule,
            future_context,
        )
        future_features = create_future_feature_rows(
            dataframe,
            schedule,
            prepared_context,
        )

    prediction_rows: list[dict[str, object]] = []
    diagnostics_rows: list[dict[str, object]] = []
    mlp_selection = pd.DataFrame()

    if strategy == "media_historica":
        if history_window is None:
            raise ValueError("A média histórica exige uma janela definida.")

        for series, group in dataframe.groupby("serie", sort=True):
            ordered = group.sort_values("data_venda")
            recent = ordered.tail(history_window).copy()
            fair = str(recent["feira"].iloc[-1])
            category = str(recent["categoria"].iloc[-1])
            date_row = schedule.loc[schedule["feira"] == fair].iloc[0]
            raw_forecast = float(recent["demanda_observada"].mean())

            prediction_row: dict[str, object] = {
                "serie": series,
                "feira": fair,
                "categoria": category,
                "data_producao_prevista": date_row[
                    "data_producao_prevista"
                ],
                "data_venda_prevista": date_row["data_venda_prevista"],
                "quantidade_observacoes_historicas": len(ordered),
                "previsao_bruta": raw_forecast,
                "alpha_suavizacao": float("nan"),
                "configuracao_selecionada": "",
                "clima": pd.NA,
                "eh_feriado": pd.NA,
                "nome_feriado": "",
                "fonte_contexto_futuro": "não requerido pelo modelo",
            }

            for lag, (_, historical_row) in enumerate(
                recent.iloc[::-1].iterrows(), start=1
            ):
                prediction_row[f"data_historico_{lag}"] = (
                    historical_row["data_venda"]
                )
                prediction_row[f"demanda_historico_{lag}"] = float(
                    historical_row["demanda_observada"]
                )

            prediction_rows.append(prediction_row)

        diagnostics_rows.append(
            {
                "modelo": model_key,
                "estrategia": strategy,
                "linhas_treino_modelo": 0,
                "configuracao_selecionada": "",
                "iteracoes_modelo_final": pd.NA,
                "perda_modelo_final": float("nan"),
                "alerta_convergencia": 0,
            }
        )

    elif strategy == "suavizacao_exponencial":
        for series, group in dataframe.groupby("serie", sort=True):
            ordered = group.sort_values("data_venda")
            fair = str(ordered["feira"].iloc[-1])
            category = str(ordered["categoria"].iloc[-1])
            date_row = schedule.loc[schedule["feira"] == fair].iloc[0]
            values = ordered["demanda_observada"].astype(float).tolist()
            alpha, training_mse = select_exponential_alpha(values)
            raw_forecast = simple_exponential_forecast(values, alpha)

            prediction_rows.append(
                {
                    "serie": series,
                    "feira": fair,
                    "categoria": category,
                    "data_producao_prevista": date_row[
                        "data_producao_prevista"
                    ],
                    "data_venda_prevista": date_row[
                        "data_venda_prevista"
                    ],
                    "quantidade_observacoes_historicas": len(ordered),
                    "previsao_bruta": raw_forecast,
                    "alpha_suavizacao": alpha,
                    "mse_treino_selecao_alpha": training_mse,
                    "configuracao_selecionada": "",
                    "clima": pd.NA,
                    "eh_feriado": pd.NA,
                    "nome_feriado": "",
                    "fonte_contexto_futuro": "não requerido pelo modelo",
                }
            )

        diagnostics_rows.append(
            {
                "modelo": model_key,
                "estrategia": strategy,
                "linhas_treino_modelo": len(dataframe),
                "configuracao_selecionada": (
                    "alpha selecionado separadamente por série"
                ),
                "iteracoes_modelo_final": pd.NA,
                "perda_modelo_final": float("nan"),
                "alerta_convergencia": 0,
            }
        )

    elif strategy in {"regressao_global", "mlp_global"}:
        training = prepare_machine_learning_training_data(dataframe)

        if strategy == "regressao_global":
            final_model = create_linear_regression_pipeline()
            final_model.fit(
                training[REGRESSION_FEATURES],
                training[TARGET_COLUMN],
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "Found unknown categories in columns .* during "
                        "transform.*"
                    ),
                    category=UserWarning,
                )
                raw_predictions = final_model.predict(
                    future_features[REGRESSION_FEATURES]
                )
            selected_name = "regressao_linear_sem_hiperparametros"
            diagnostics = {
                "iteracoes": pd.NA,
                "perda_final": float("nan"),
                "alerta_convergencia": 0,
            }
        else:
            selected_configuration, mlp_selection = (
                select_mlp_configuration(training)
            )
            final_model, diagnostics = fit_mlp_model(
                training,
                selected_configuration,
            )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message=(
                        "Found unknown categories in columns .* during "
                        "transform.*"
                    ),
                    category=UserWarning,
                )
                raw_predictions = final_model.predict(
                    future_features[REGRESSION_FEATURES]
                )
            selected_name = str(
                selected_configuration["configuracao"]
            )

        for (_, future_row), raw_prediction in zip(
            future_features.iterrows(), raw_predictions
        ):
            prediction_rows.append(
                {
                    **future_row.to_dict(),
                    "previsao_bruta": float(raw_prediction),
                    "alpha_suavizacao": float("nan"),
                    "configuracao_selecionada": selected_name,
                }
            )

        diagnostics_rows.append(
            {
                "modelo": model_key,
                "estrategia": strategy,
                "linhas_treino_modelo": len(training),
                "configuracao_selecionada": selected_name,
                "iteracoes_modelo_final": diagnostics["iteracoes"],
                "perda_modelo_final": diagnostics["perda_final"],
                "alerta_convergencia": diagnostics[
                    "alerta_convergencia"
                ],
            }
        )

    else:
        raise ValueError(
            f"Estratégia operacional desconhecida: {strategy}."
        )

    predictions = pd.DataFrame(prediction_rows)
    predictions["previsao_modelo_sem_limite"] = pd.to_numeric(
        predictions["previsao_bruta"], errors="raise"
    )
    predictions["ajustada_para_zero"] = predictions[
        "previsao_modelo_sem_limite"
    ].lt(0).astype(int)
    predictions["previsao_bruta"] = predictions[
        "previsao_modelo_sem_limite"
    ].clip(lower=0.0)
    predictions["previsao_operacional"] = predictions[
        "previsao_bruta"
    ].map(_round_half_up)
    predictions["data_corte"] = effective_cutoff
    predictions["modelo"] = model_key
    predictions["nome_modelo"] = model_name
    predictions["estrategia_operacional"] = strategy
    predictions["janela_historica"] = (
        history_window if history_window is not None else "todo_historico"
    )
    predictions["metodo_arredondamento"] = (
        "inteiro mais próximo; meio para cima"
    )
    predictions["status"] = (
        "previsão de demanda; não é recomendação de produção"
    )

    audit_window = (
        history_window if strategy == "media_historica" else None
    )
    history = create_history_audit(dataframe, audit_window)
    history["data_corte"] = effective_cutoff
    history["modelo"] = model_key

    fair_rank = {fair: position for position, fair in enumerate(FAIR_ORDER)}
    predictions["_ordem_feira"] = predictions["feira"].map(fair_rank)
    predictions = predictions.sort_values(
        ["_ordem_feira", "categoria"]
    ).drop(columns="_ordem_feira").reset_index(drop=True)

    history["_ordem_feira"] = history["feira"].map(fair_rank)
    history = history.sort_values(
        ["_ordem_feira", "categoria", "posicao_na_janela"]
    ).drop(columns="_ordem_feira").reset_index(drop=True)

    artifacts = {
        "contexto_futuro_previsao": prepared_context,
        "selecao_mlp_operacional": mlp_selection,
        "diagnostico_modelo_operacional": pd.DataFrame(diagnostics_rows),
    }

    return predictions, history, effective_cutoff, artifacts


def create_operational_summary(
    predictions: pd.DataFrame,
) -> pd.DataFrame:
    """Resume as previsões por feira e no total geral."""

    summary = (
        predictions.groupby(
            [
                "feira",
                "data_producao_prevista",
                "data_venda_prevista",
            ],
            as_index=False,
        )
        .agg(
            quantidade_categorias=("categoria", "nunique"),
            previsao_total_bruta=("previsao_bruta", "sum"),
            previsao_total_operacional=("previsao_operacional", "sum"),
        )
    )
    summary["previsao_total_bruta"] = summary[
        "previsao_total_bruta"
    ].round(4)
    summary["diferenca_arredondamento_total"] = (
        summary["previsao_total_operacional"]
        - summary["previsao_total_bruta"]
    ).round(4)

    fair_rank = {fair: position for position, fair in enumerate(FAIR_ORDER)}
    summary["_ordem_feira"] = summary["feira"].map(fair_rank)
    return summary.sort_values("_ordem_feira").drop(
        columns="_ordem_feira"
    ).reset_index(drop=True)


def create_operational_decision(
    phase10_analyses: dict[str, pd.DataFrame],
    predictions: pd.DataFrame,
    source_record_count: int,
    model_key: str,
    model_name: str,
    history_window: int | None,
    strategy: str,
    diagnostics: pd.DataFrame,
) -> pd.DataFrame:
    """Registra o modelo revalidado e o snapshot usado na previsão."""

    decision = phase10_analyses["decisao_modelo_fase10"].iloc[0]
    diagnostic = diagnostics.iloc[0]
    return pd.DataFrame(
        [
            {
                "modelo_operacional": model_key,
                "nome_modelo_operacional": model_name,
                "criterio_selecao": "menor MAE no walk-forward",
                "mae_validacao": float(decision["mae_lider"]),
                "mse_validacao": float(decision["mse_lider"]),
                "rmse_validacao": float(decision["rmse_lider"]),
                "mape_validacao": float(decision["mape_lider"]),
                "registros_origem_mysql": int(source_record_count),
                "data_corte": predictions["data_corte"].max(),
                "quantidade_series_previstas": len(predictions),
                "estrategia_operacional": strategy,
                "janela_historica": (
                    history_window
                    if history_window is not None
                    else "todo_historico"
                ),
                "linhas_treino_modelo_final": int(
                    diagnostic["linhas_treino_modelo"]
                ),
                "configuracao_operacional_selecionada": diagnostic[
                    "configuracao_selecionada"
                ],
                "contexto_futuro_requerido": (
                    "sim"
                    if SUPPORTED_OPERATIONAL_MODELS[model_key][
                        "requer_contexto_futuro"
                    ]
                    else "não"
                ),
                "recomendacao_producao_gerada": "não",
                "status": (
                    "Modelo revalidado com a base atual e aplicado a todo "
                    "o histórico disponível."
                ),
            }
        ]
    )


def create_operational_configuration(
    model_name: str,
    history_window: int | None,
    strategy: str,
    requires_future_context: bool,
) -> pd.DataFrame:
    """Documenta as escolhas da previsão operacional."""

    rows = [
        (
            "granularidade",
            "categoria + feira",
            "Mantém o nível definido na Fase 5.",
        ),
        (
            "modelo",
            model_name,
            "Obteve o menor MAE na comparação temporal da Fase 10.",
        ),
        (
            "janela",
            (
                str(history_window)
                if history_window is not None
                else "todo o histórico disponível"
            ),
            (
                "Os atributos históricos usam somente observações anteriores "
                "à data prevista."
            ),
        ),
        (
            "estrategia_operacional",
            strategy,
            "A rotina é escolhida automaticamente pelo modelo líder.",
        ),
        (
            "contexto_futuro",
            (
                "clima e feriado informados no terminal"
                if requires_future_context
                else "não requerido pelo modelo"
            ),
            (
                "Valores futuros não são inventados nem obtidos das "
                "observações posteriores."
            ),
        ),
        (
            "horizonte",
            "próxima ocorrência de cada feira",
            "Corresponde ao horizonte de um passo avaliado no walk-forward.",
        ),
        (
            "data_corte",
            "maior data de venda disponível",
            "Impede o uso de observações posteriores ao snapshot.",
        ),
        (
            "arredondamento",
            "inteiro mais próximo; meio para cima",
            "A demanda é expressa em unidades inteiras, sem criar margem extra.",
        ),
        (
            "revalidacao_modelo",
            "ranking integral recalculado antes da previsão",
            "Evita aplicar uma decisão antiga a uma versão diferente do banco.",
        ),
        (
            "escopo",
            "somente previsão de demanda",
            "Distribuição por produto e otimização serão fases posteriores.",
        ),
    ]

    return pd.DataFrame(
        rows,
        columns=["parametro", "valor", "justificativa"],
    )


def validate_operational_forecast(
    predictions: pd.DataFrame,
    history: pd.DataFrame,
    phase10_analyses: dict[str, pd.DataFrame],
    model_key: str,
    history_window: int | None,
    strategy: str,
    requires_future_context: bool,
    artifacts: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Audita as previsões futuras e o histórico utilizado."""

    checks: list[dict[str, str]] = []

    def add_check(name: str, passed: bool, details: str) -> None:
        checks.append(
            {
                "teste": name,
                "resultado": "OK" if passed else "FALHA",
                "detalhes": details,
            }
        )

    series_count = predictions["serie"].nunique()
    add_check(
        "Previsão para as 18 séries",
        len(predictions) == EXPECTED_SERIES
        and series_count == EXPECTED_SERIES,
        f"Linhas: {len(predictions)}; séries: {series_count}",
    )

    history_counts = history.groupby("serie").size()
    if strategy == "media_historica":
        expected_history = history_window
        history_coverage_ok = (
            expected_history is not None
            and len(history_counts) == EXPECTED_SERIES
            and history_counts.eq(expected_history).all()
        )
        history_description = (
            f"Janela esperada: {expected_history}; contagens: "
            f"{sorted(history_counts.unique().tolist())}"
        )
    else:
        history_coverage_ok = (
            len(history_counts) == EXPECTED_SERIES
            and history_counts.ge(4).all()
        )
        history_description = (
            "Todo o histórico disponível por série; contagens: "
            f"{sorted(history_counts.unique().tolist())}"
        )
    add_check(
        "Cobertura histórica adequada ao modelo",
        bool(history_coverage_ok),
        history_description,
    )

    duplicated = predictions.duplicated(
        subset=["data_venda_prevista", "feira", "categoria"]
    ).sum()
    add_check(
        "Chave futura sem duplicidades",
        duplicated == 0,
        f"Duplicidades encontradas: {int(duplicated)}",
    )

    history_before_cutoff = (
        history["data_venda_historica"] <= history["data_corte"]
    ).all()
    add_check(
        "Histórico limitado à data de corte",
        bool(history_before_cutoff),
        "Todas as observações usadas devem existir na data da previsão.",
    )

    future_after_cutoff = (
        predictions["data_venda_prevista"] > predictions["data_corte"]
    ).all()
    add_check(
        "Datas previstas posteriores à data de corte",
        bool(future_after_cutoff),
        "A previsão deve se referir a uma ocorrência ainda não observada.",
    )

    weekdays_ok = all(
        row.data_venda_prevista.weekday() == FAIR_WEEKDAYS[row.feira]
        for row in predictions.itertuples()
    )
    add_check(
        "Calendário semanal das feiras respeitado",
        weekdays_ok,
        "QUA=quarta, QUI=quinta, SAB_*=sábado e DOM_*=domingo.",
    )

    production_dates_ok = (
        predictions["data_producao_prevista"]
        == predictions["data_venda_prevista"] - pd.Timedelta(days=1)
    ).all()
    add_check(
        "Data de produção anterior à venda",
        bool(production_dates_ok),
        "A produção deve ocorrer um dia antes da feira.",
    )

    forecast_by_series = predictions.set_index("serie")

    if strategy == "media_historica":
        expected_forecasts = history.groupby("serie")[
            "demanda_observada"
        ].mean()
        calculation_ok = all(
            abs(
                float(forecast_by_series.loc[series, "previsao_bruta"])
                - float(value)
            )
            <= 0.0001
            for series, value in expected_forecasts.items()
        )
        calculation_details = (
            f"Média das {history_window} ocorrências anteriores."
        )
    elif strategy == "suavizacao_exponencial":
        calculation_ok = True
        for series, series_history in history.groupby("serie"):
            values = series_history.sort_values(
                "data_venda_historica"
            )["demanda_observada"].astype(float).tolist()
            alpha = float(
                forecast_by_series.loc[series, "alpha_suavizacao"]
            )
            expected = simple_exponential_forecast(values, alpha)
            found = float(
                forecast_by_series.loc[series, "previsao_bruta"]
            )
            if abs(expected - found) > 0.0001:
                calculation_ok = False
                break
        calculation_details = (
            "Alpha selecionado e suavização recalculados somente com o "
            "histórico de cada série."
        )
    else:
        diagnostics = artifacts["diagnostico_modelo_operacional"]
        calculation_ok = (
            len(diagnostics) == 1
            and int(diagnostics["linhas_treino_modelo"].iloc[0]) > 0
            and str(
                diagnostics["configuracao_selecionada"].iloc[0]
            ).strip()
            != ""
        )
        calculation_details = (
            "Modelo global reajustado com todo o dataset histórico de "
            "modelagem."
        )
    add_check(
        "Cálculo operacional compatível com o modelo líder",
        bool(calculation_ok),
        calculation_details,
    )

    finite_forecasts = all(
        isfinite(float(value))
        for value in predictions[
            ["previsao_modelo_sem_limite", "previsao_bruta"]
        ].stack().tolist()
    )
    nonnegative = predictions[
        ["previsao_bruta", "previsao_operacional"]
    ].ge(0).all().all()
    add_check(
        "Previsões válidas e não negativas",
        finite_forecasts and bool(nonnegative),
        "Não são permitidos valores ausentes, infinitos ou negativos.",
    )

    rounded_ok = all(
        int(value) == value
        for value in predictions["previsao_operacional"].tolist()
    )
    add_check(
        "Previsões operacionais inteiras",
        rounded_ok,
        "O valor operacional deve representar unidades inteiras.",
    )

    categories_per_fair = predictions.groupby("feira")["categoria"].nunique()
    add_check(
        "Três categorias por feira",
        len(categories_per_fair) == len(FAIR_WEEKDAYS)
        and categories_per_fair.eq(3).all(),
        f"Categorias por feira: {categories_per_fair.to_dict()}",
    )

    same_date_per_fair = predictions.groupby("feira")[
        "data_venda_prevista"
    ].nunique().eq(1).all()
    add_check(
        "Data futura única por feira",
        bool(same_date_per_fair),
        "As três categorias da mesma feira devem compartilhar a data de venda.",
    )

    models_ok = predictions["modelo"].eq(model_key).all()
    add_check(
        "Modelo operacional identificado",
        bool(models_ok),
        f"Modelo esperado: {model_key}",
    )

    decision = phase10_analyses["decisao_modelo_fase10"].iloc[0]
    current_leader = str(decision["modelo_lider_fase10"])
    add_check(
        "Modelo líder revalidado com a base atual",
        current_leader == model_key,
        f"Líder encontrado: {current_leader}",
    )

    context = artifacts["contexto_futuro_previsao"]
    if requires_future_context:
        context_ok = (
            len(context) == len(FAIR_ORDER)
            and set(context["feira"]) == set(FAIR_ORDER)
            and context["clima"].isin(CLIMATE_OPTIONS).all()
            and context["eh_feriado"].isin([0, 1]).all()
        )
        prediction_context_ok = True
        for fair, group in predictions.groupby("feira"):
            context_row = context.loc[context["feira"] == fair].iloc[0]
            if (
                not group["clima"].eq(context_row["clima"]).all()
                or not group["eh_feriado"].eq(
                    context_row["eh_feriado"]
                ).all()
            ):
                prediction_context_ok = False
                break
    else:
        context_ok = context.empty
        prediction_context_ok = True
    add_check(
        "Contexto futuro tratado explicitamente",
        bool(context_ok and prediction_context_ok),
        (
            "Clima e feriado foram informados para as seis feiras."
            if requires_future_context
            else "O modelo líder não utiliza clima ou feriado futuro."
        ),
    )

    if strategy in {"regressao_global", "mlp_global"}:
        lags_ok = True
        for series, series_history in history.groupby("serie"):
            latest = series_history.sort_values(
                "data_venda_historica"
            )["demanda_observada"].astype(float).tail(3).tolist()
            predicted_row = forecast_by_series.loc[series]
            expected_lags = [latest[-1], latest[-2], latest[-3]]
            found_lags = [
                float(predicted_row[f"lag_{lag}"])
                for lag in [1, 2, 3]
            ]
            if any(
                abs(expected - found) > 0.0001
                for expected, found in zip(expected_lags, found_lags)
            ):
                lags_ok = False
                break
    else:
        lags_ok = True
    add_check(
        "Atributos futuros usam somente histórico anterior",
        bool(lags_ok),
        "lag_1, lag_2 e lag_3 foram conferidos contra o histórico.",
    )

    diagnostics = artifacts["diagnostico_modelo_operacional"]
    diagnostics_ok = (
        len(diagnostics) == 1
        and diagnostics["modelo"].eq(model_key).all()
    )
    if strategy == "mlp_global":
        diagnostics_ok = diagnostics_ok and diagnostics[
            "alerta_convergencia"
        ].eq(0).all()
    add_check(
        "Treinamento operacional documentado",
        bool(diagnostics_ok),
        (
            "Convergência final monitorada e configuração registrada."
            if strategy == "mlp_global"
            else "Estratégia e volume de treinamento registrados."
        ),
    )

    selection = artifacts["selecao_mlp_operacional"]
    if strategy == "mlp_global":
        selection_ok = (
            len(selection) == 4
            and int(selection["selecionada"].sum()) == 1
            and selection["ordem_temporal_interna_ok"].all()
            and selection["series_validacao_interna"].eq(
                EXPECTED_SERIES
            ).all()
        )
    else:
        selection_ok = selection.empty
    add_check(
        "Seleção operacional da MLP sem vazamento temporal",
        bool(selection_ok),
        (
            "Quatro configurações avaliadas com a última observação de "
            "cada série reservada para validação interna."
            if strategy == "mlp_global"
            else "Verificação não aplicável ao modelo líder atual."
        ),
    )

    return pd.DataFrame(checks)


def run_operational_forecast(
    category_fair_data: pd.DataFrame,
    phase10_analyses: dict[str, pd.DataFrame],
    source_record_count: int,
    cutoff_date: str | pd.Timestamp | None = None,
    future_context: pd.DataFrame | None = None,
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], None] = print,
) -> dict[str, pd.DataFrame]:
    """Executa a previsão operacional da Fase 11."""

    (
        model_key,
        model_name,
        history_window,
        strategy,
        requires_future_context,
    ) = get_operational_model(phase10_analyses)

    predictions, history, _, artifacts = create_operational_predictions(
        category_fair_data,
        model_key,
        model_name,
        history_window,
        strategy,
        requires_future_context,
        cutoff_date=cutoff_date,
        future_context=future_context,
        input_function=input_function,
        output_function=output_function,
    )
    decision = create_operational_decision(
        phase10_analyses,
        predictions,
        source_record_count,
        model_key,
        model_name,
        history_window,
        strategy,
        artifacts["diagnostico_modelo_operacional"],
    )
    validation = validate_operational_forecast(
        predictions,
        history,
        phase10_analyses,
        model_key,
        history_window,
        strategy,
        requires_future_context,
        artifacts,
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
            "Falha na previsão operacional: " + "; ".join(failed)
        )

    return {
        "previsoes_operacionais_categoria_feira": predictions,
        "historico_previsao_operacional": history,
        "resumo_previsao_operacional": create_operational_summary(
            predictions
        ),
        "validacao_previsao_operacional": validation,
        "configuracao_previsao_operacional": (
            create_operational_configuration(
                model_name,
                history_window,
                strategy,
                requires_future_context,
            )
        ),
        "decisao_modelo_operacional": decision,
        **artifacts,
    }


def save_operational_forecast_outputs(
    analyses: dict[str, pd.DataFrame],
) -> list[Path]:
    """Salva previsões, auditorias e decisões da Fase 11."""

    project_root = Path(__file__).resolve().parent.parent
    forecasts_directory = project_root / "reports" / "forecasts"
    tables_directory = project_root / "reports" / "tables"

    forecasts_directory.mkdir(parents=True, exist_ok=True)
    tables_directory.mkdir(parents=True, exist_ok=True)

    destinations = {
        "previsoes_operacionais_categoria_feira": forecasts_directory,
        "historico_previsao_operacional": forecasts_directory,
        "resumo_previsao_operacional": tables_directory,
        "validacao_previsao_operacional": tables_directory,
        "configuracao_previsao_operacional": tables_directory,
        "decisao_modelo_operacional": tables_directory,
        "contexto_futuro_previsao": forecasts_directory,
        "selecao_mlp_operacional": tables_directory,
        "diagnostico_modelo_operacional": tables_directory,
    }

    generated_files = []

    for name, dataframe in analyses.items():
        path = destinations[name] / f"{name}.csv"
        dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        generated_files.append(path)

    return generated_files
