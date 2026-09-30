"""
Previsão operacional.

O modelo selecionado nas fases de avaliação é reaplicado a todo o histórico
disponível para gerar uma previsão de demanda para a próxima ocorrência de
cada combinação categoria + feira. Esta fase gera previsões, não recomendações
de produção e não distribui quantidades entre produtos.
"""

from decimal import Decimal, ROUND_HALF_UP
from math import isfinite
from pathlib import Path

import pandas as pd


EXPECTED_SERIES = 18

SUPPORTED_OPERATIONAL_MODELS = {
    "naive": {
        "nome": "Naive (última ocorrência)",
        "janela": 1,
    },
    "media_movel_2": {
        "nome": "Média móvel (2 ocorrências)",
        "janela": 2,
    },
    "media_movel_3": {
        "nome": "Média móvel (3 ocorrências)",
        "janela": 3,
    },
    "media_movel_4": {
        "nome": "Média móvel (4 ocorrências)",
        "janela": 4,
    },
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
) -> tuple[str, str, int]:
    """Obtém o modelo vencedor e sua janela operacional revalidada."""

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
    history_window = int(configuration["janela"])

    return model_key, model_name, history_window


def prepare_operational_history(
    category_fair_data: pd.DataFrame,
    history_window: int,
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

    dataframe = category_fair_data[required_columns].copy()
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
    insufficient = counts.loc[counts < history_window]

    if not insufficient.empty:
        raise ValueError(
            "Histórico insuficiente para o modelo operacional com janela "
            f"de {history_window} ocorrência(s): "
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


def create_operational_predictions(
    category_fair_data: pd.DataFrame,
    model_key: str,
    model_name: str,
    history_window: int,
    cutoff_date: str | pd.Timestamp | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Timestamp]:
    """Gera uma previsão para a próxima ocorrência de cada série."""

    dataframe, effective_cutoff = prepare_operational_history(
        category_fair_data,
        history_window,
        cutoff_date,
    )

    prediction_rows: list[dict[str, object]] = []
    history_rows: list[dict[str, object]] = []

    for series, group in dataframe.groupby("serie", sort=True):
        ordered = group.sort_values("data_venda")
        recent = ordered.tail(history_window).copy()
        fair = str(recent["feira"].iloc[-1])
        category = str(recent["categoria"].iloc[-1])

        sale_date = _next_weekday_after(
            effective_cutoff,
            FAIR_WEEKDAYS[fair],
        )
        production_date = sale_date - pd.Timedelta(days=1)

        raw_forecast = float(recent["demanda_observada"].mean())
        operational_forecast = _round_half_up(raw_forecast)

        history_values = recent["demanda_observada"].astype(float).tolist()
        history_dates = recent["data_venda"].tolist()

        for position, (_, historical_row) in enumerate(
            recent.iterrows(), start=1
        ):
            history_rows.append(
                {
                    "data_corte": effective_cutoff,
                    "serie": series,
                    "feira": fair,
                    "categoria": category,
                    "posicao_na_janela": position,
                    "defasagem": history_window - position + 1,
                    "data_venda_historica": historical_row["data_venda"],
                    "demanda_observada": float(
                        historical_row["demanda_observada"]
                    ),
                    "utilizada_na_previsao": 1,
                }
            )

        prediction_row: dict[str, object] = {
                "data_corte": effective_cutoff,
                "data_producao_prevista": production_date,
                "data_venda_prevista": sale_date,
                "serie": series,
                "feira": fair,
                "categoria": category,
                "modelo": model_key,
                "nome_modelo": model_name,
                "janela_historica": history_window,
                "quantidade_observacoes_historicas": len(ordered),
                # Mantém a precisão integral para que somatórios posteriores
                # não acumulem o arredondamento individual das categorias.
                "previsao_bruta": raw_forecast,
                "previsao_operacional": operational_forecast,
                "metodo_arredondamento": "inteiro mais próximo; meio para cima",
                "status": "previsão de demanda; não é recomendação de produção",
            }

        for position, (history_date, history_value) in enumerate(
            zip(history_dates, history_values), start=1
        ):
            lag = history_window - position + 1
            prediction_row[f"data_historico_{lag}"] = history_date
            prediction_row[f"demanda_historico_{lag}"] = history_value

        prediction_rows.append(prediction_row)

    predictions = pd.DataFrame(prediction_rows)
    history = pd.DataFrame(history_rows)

    fair_rank = {fair: position for position, fair in enumerate(FAIR_ORDER)}
    predictions["_ordem_feira"] = predictions["feira"].map(fair_rank)
    predictions = predictions.sort_values(
        ["_ordem_feira", "categoria"]
    ).drop(columns="_ordem_feira").reset_index(drop=True)

    history["_ordem_feira"] = history["feira"].map(fair_rank)
    history = history.sort_values(
        ["_ordem_feira", "categoria", "posicao_na_janela"]
    ).drop(columns="_ordem_feira").reset_index(drop=True)

    return predictions, history, effective_cutoff


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
    history_window: int,
) -> pd.DataFrame:
    """Registra o modelo revalidado e o snapshot usado na previsão."""

    decision = phase10_analyses["decisao_modelo_fase10"].iloc[0]
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
                "janela_historica": history_window,
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
    history_window: int,
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
            str(history_window),
            (
                "Utiliza somente as ocorrências anteriores da mesma série, "
                "conforme a janela do modelo revalidado."
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
    history_window: int,
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
    add_check(
        f"{history_window} observação(ões) histórica(s) por série",
        len(history_counts) == EXPECTED_SERIES
        and history_counts.eq(history_window).all(),
        f"Contagens encontradas: {sorted(history_counts.unique().tolist())}",
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

    expected_means = history.groupby("serie")["demanda_observada"].mean()
    forecast_by_series = predictions.set_index("serie")["previsao_bruta"]
    means_ok = all(
        abs(float(forecast_by_series[series]) - float(value)) <= 0.0001
        for series, value in expected_means.items()
    )
    add_check(
        "Média móvel calculada corretamente",
        means_ok,
        (
            "A previsão bruta deve ser a média das "
            f"{history_window} ocorrências registradas."
        ),
    )

    finite_forecasts = all(
        isfinite(float(value))
        for value in predictions["previsao_bruta"].tolist()
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

    return pd.DataFrame(checks)


def run_operational_forecast(
    category_fair_data: pd.DataFrame,
    phase10_analyses: dict[str, pd.DataFrame],
    source_record_count: int,
    cutoff_date: str | pd.Timestamp | None = None,
) -> dict[str, pd.DataFrame]:
    """Executa a previsão operacional da Fase 11."""

    model_key, model_name, history_window = get_operational_model(
        phase10_analyses
    )

    predictions, history, _ = create_operational_predictions(
        category_fair_data,
        model_key,
        model_name,
        history_window,
        cutoff_date,
    )
    decision = create_operational_decision(
        phase10_analyses,
        predictions,
        source_record_count,
        model_key,
        model_name,
        history_window,
    )
    validation = validate_operational_forecast(
        predictions,
        history,
        phase10_analyses,
        model_key,
        history_window,
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
            create_operational_configuration(model_name, history_window)
        ),
        "decisao_modelo_operacional": decision,
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
    }

    generated_files = []

    for name, dataframe in analyses.items():
        path = destinations[name] / f"{name}.csv"
        dataframe.to_csv(path, index=False, encoding="utf-8-sig")
        generated_files.append(path)

    return generated_files
