"""Auditoria diagnóstica das previsões; não escolhe modelos nem altera valores.

Estar fora da faixa histórica é um sinal para revisão, não uma prova de erro,
um intervalo probabilístico ou uma restrição de produção.
"""

from math import isclose, isfinite, sqrt
from pathlib import Path

import pandas as pd

from forecasting.statistical_models import REGRESSION_FEATURES
from preprocessing.features import HISTORICAL_FEATURES


def _require(frame: pd.DataFrame, columns: list[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing or frame.empty:
        raise ValueError(f"{name}: dados vazios ou colunas ausentes: {missing}")


def _numeric(frame: pd.DataFrame, columns: list[str]) -> None:
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if not frame[column].map(lambda value: isfinite(float(value))).all():
            raise ValueError(f"Valores ausentes ou não finitos em {column}.")
        if frame[column].lt(0).any():
            raise ValueError(f"Valores negativos em {column}.")


def compare_models_by_series(
    comparison: pd.DataFrame, cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Recalcula erros sem usar métricas previamente arredondadas."""
    required = ["modelo", "nome_modelo", "serie", "feira", "categoria",
                "fold", "id_linha_modelagem", "data_venda", "valor_real",
                "previsao"]
    _require(comparison, required, "Previsões walk-forward")
    data = comparison[required].copy()
    _numeric(data, ["valor_real", "previsao"])
    data["data_venda"] = pd.to_datetime(data["data_venda"], errors="raise")
    if data["data_venda"].isna().any() or data["data_venda"].gt(cutoff).any():
        raise ValueError("Casos de avaliação posteriores ao corte ou sem data.")
    case_key = ["fold", "id_linha_modelagem"]
    if data.duplicated(["modelo", *case_key]).any():
        raise ValueError("Casos de avaliação duplicados por modelo.")
    expected = None
    for _, group in data.groupby("modelo"):
        cases = set(group[case_key].itertuples(index=False, name=None))
        if expected is not None and cases != expected:
            raise ValueError("Os modelos não possuem os mesmos casos de teste.")
        expected = cases
    metadata = ["serie", "feira", "categoria", "data_venda", "valor_real"]
    if data.groupby(case_key)[metadata].nunique(dropna=False).gt(1).any().any():
        raise ValueError("Alvos ou identificação dos casos divergem entre modelos.")

    rows = []
    keys = ["modelo", "nome_modelo", "serie", "feira", "categoria"]
    for values, group in data.groupby(keys, sort=True):
        error = group["previsao"] - group["valor_real"]
        absolute = error.abs()
        nonzero = group["valor_real"].ne(0)
        total = float(group["valor_real"].sum())
        rows.append({
            **dict(zip(keys, values)), "data_corte": cutoff,
            "quantidade_previsoes": len(group),
            "quantidade_alvos_zero": int((~nonzero).sum()),
            "mae": float(absolute.mean()),
            "rmse": sqrt(float((error ** 2).mean())),
            "mape": float((absolute[nonzero] / group.loc[nonzero, "valor_real"] * 100).mean()),
            "wape": float(absolute.sum() / total * 100) if total else float("nan"),
            "vies_medio_previsto_menos_real": float(error.mean()),
        })
    metrics = pd.DataFrame(rows).sort_values(
        ["serie", "mae", "rmse", "mape", "modelo"], na_position="last",
    ).reset_index(drop=True)
    metrics["posicao_na_serie"] = metrics.groupby("serie").cumcount() + 1
    metrics["uso"] = "comparação diagnóstica; não muda o modelo operacional"
    return metrics


def audit_future_attributes(
    history: pd.DataFrame, operational: dict[str, pd.DataFrame],
    predictions: pd.DataFrame, cutoff: pd.Timestamp,
) -> pd.DataFrame:
    """Confere a entrada efetivamente enviada ao modelo global."""
    columns = ["data_corte", "serie", "modelo", "atributo", "valor_futuro",
               "categorias_vistas_treino",
               "valor_esperado", "minimo_treino_serie", "maximo_treino_serie",
               "fora_faixa_treino", "verificacao", "observacao"]
    machine_learning = predictions["modelo"].isin(
        ["mlp_global", "regressao_linear_global"]
    ).any()
    if not machine_learning:
        return pd.DataFrame(columns=columns)
    future = operational.get("atributos_futuros_previsao", pd.DataFrame())
    _require(future, ["serie", "data_venda_prevista", *REGRESSION_FEATURES],
             "Atributos futuros efetivos")
    if future["serie"].duplicated().any() or set(future["serie"]) != set(predictions["serie"]):
        raise ValueError("Cobertura dos atributos futuros inconsistente.")
    dates = pd.to_datetime(future["data_venda_prevista"], errors="raise")
    if dates.isna().any() or not dates.gt(cutoff).all():
        raise ValueError("Datas dos atributos futuros devem ser posteriores ao corte.")
    _require(history, [*HISTORICAL_FEATURES, *REGRESSION_FEATURES], "Histórico de atributos")
    training = history.dropna(subset=HISTORICAL_FEATURES)
    if training.empty:
        raise ValueError("Histórico de treino da auditoria vazio.")
    rows = []
    for _, row in future.iterrows():
        group = history.loc[history["serie"].eq(row["serie"])].sort_values("data_venda")
        values = group["demanda_observada"].astype(float).tolist()
        expected = {f"lag_{lag}": values[-lag] for lag in (1, 2, 3)}
        expected.update({f"media_movel_{window}": sum(values[-window:]) / window
                         for window in (2, 3, 4)})
        expected["tendencia_serie"] = len(group) + 1
        current_train = training.loc[training["serie"].eq(row["serie"])]
        for attribute in ["tendencia_serie", *HISTORICAL_FEATURES, "eh_feriado",
                          "feira", "categoria", "clima"]:
            value = row[attribute]
            checked = attribute in expected
            valid = (not checked or (isfinite(float(value)) and isclose(
                float(value), expected[attribute], rel_tol=1e-9, abs_tol=1e-9)))
            if not valid:
                raise ValueError(f"Atributo {attribute} inconsistente na série {row['serie']}.")
            categorical = attribute in {"feira", "categoria", "clima"}
            if categorical:
                outside = value not in set(training[attribute].dropna())
                minimum = maximum = pd.NA
            else:
                minimum = current_train[attribute].min()
                maximum = current_train[attribute].max()
                outside = bool(float(value) < minimum or float(value) > maximum)
            rows.append({
                "data_corte": cutoff, "serie": row["serie"],
                "modelo": predictions["modelo"].iloc[0], "atributo": attribute,
                "valor_futuro": value, "valor_esperado": expected.get(attribute, pd.NA),
                "categorias_vistas_treino": ", ".join(sorted(str(v) for v in training[attribute].dropna().unique())) if categorical else "",
                "minimo_treino_serie": minimum, "maximo_treino_serie": maximum,
                "fora_faixa_treino": int(outside),
                "verificacao": "OK" if checked else "CONTEXTO_REGISTRADO",
                "observacao": (
                    "Avanço temporal esperado; não indica bug por si só."
                    if attribute == "tendencia_serie" else
                    "Categoria não vista no treino global." if categorical and outside else
                    "Valor fora da faixa empírica do treino desta série." if outside else ""
                ),
            })
    return pd.DataFrame(rows, columns=columns)


def run_forecast_audit(
    category_fair_data: pd.DataFrame,
    operational_analyses: dict[str, pd.DataFrame],
    comparison_predictions: pd.DataFrame,
    source_data: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    """Audita o snapshot atual sem consultar CSVs de execuções antigas."""
    predictions = operational_analyses["previsoes_operacionais_categoria_feira"].copy()
    _require(predictions, ["serie", "feira", "categoria", "modelo", "data_corte",
                           "previsao_bruta", "previsao_operacional"], "Previsões atuais")
    _numeric(predictions, ["previsao_bruta", "previsao_operacional"])
    cuts = pd.to_datetime(predictions["data_corte"], errors="raise")
    if cuts.isna().any() or cuts.nunique() != 1 or predictions["serie"].duplicated().any():
        raise ValueError("Corte ou chaves das previsões inconsistentes.")
    cutoff = cuts.iloc[0]
    _require(category_fair_data, ["serie", "feira", "categoria", "data_venda",
                                  "demanda_observada"], "Histórico agregado")
    history = category_fair_data.copy()
    history["data_venda"] = pd.to_datetime(history["data_venda"], errors="raise")
    if history["data_venda"].isna().any():
        raise ValueError("Histórico com datas ausentes.")
    history = history.loc[history["data_venda"].le(cutoff)].sort_values(
        ["serie", "data_venda"]
    ).copy()
    _numeric(history, ["demanda_observada"])
    if history.duplicated(["serie", "data_venda"]).any():
        raise ValueError("Histórico duplicado por série e data.")
    metrics = compare_models_by_series(comparison_predictions, cutoff)
    if set(metrics["serie"]) != set(predictions["serie"]):
        raise ValueError("Séries da avaliação e da previsão divergem.")
    attributes = audit_future_attributes(history, operational_analyses, predictions, cutoff)
    rows = []
    for _, forecast in predictions.iterrows():
        group = history.loc[history["serie"].eq(forecast["serie"])]
        if len(group) < 4:
            raise ValueError(f"Histórico inferior a quatro ocorrências: {forecast['serie']}.")
        if not group["feira"].eq(forecast["feira"]).all() or not group["categoria"].eq(forecast["categoria"]).all():
            raise ValueError("Identificação da série inconsistente.")
        values = group["demanda_observada"]
        raw = float(forecast["previsao_bruta"])
        mm4 = float(values.tail(4).mean())
        high = raw > float(values.max())
        low = raw < float(values.min())
        recent_max = float(values.tail(4).max())
        series_metrics = metrics.loc[metrics["serie"].eq(forecast["serie"])]
        leader = series_metrics.iloc[0]
        selected = series_metrics.loc[series_metrics["modelo"].eq(forecast["modelo"])]
        if len(selected) != 1:
            raise ValueError("Modelo operacional ausente da comparação por série.")
        attribute_alert = attributes.loc[
            attributes["serie"].eq(forecast["serie"])
            & attributes["fora_faixa_treino"].eq(1)
            & attributes["atributo"].ne("tendencia_serie")
        ]
        reasons = []
        if high:
            reasons.append("previsão acima do máximo histórico de vendas")
        if low:
            reasons.append("previsão abaixo do mínimo histórico de vendas")
        if not high and raw > recent_max:
            reasons.append("previsão acima do máximo das últimas quatro ocorrências")
        if not attribute_alert.empty:
            reasons.append("atributos futuros fora da faixa/categorias vistas no treino")
        rows.append({
            "data_corte": cutoff, "serie": forecast["serie"],
            "feira": forecast["feira"], "categoria": forecast["categoria"],
            "modelo_operacional": forecast["modelo"],
            "previsao_bruta": raw, "previsao_operacional": forecast["previsao_operacional"],
            "observacoes_historicas": len(group),
            "primeira_data_historica": group["data_venda"].min(),
            "ultima_data_historica": group["data_venda"].max(),
            "minimo_vendido_historico": float(values.min()),
            "maximo_vendido_historico": float(values.max()),
            "media_vendida_historica": float(values.mean()),
            "mediana_vendida_historica": float(values.median()),
            "media_produzida_historica": float(group["total_produzido"].mean()) if "total_produzido" in group else float("nan"),
            "maximo_produzido_historico": float(group["total_produzido"].max()) if "total_produzido" in group else float("nan"),
            "ultima_venda": float(values.iloc[-1]),
            **{f"media_movel_{window}": float(values.tail(window).mean()) for window in (2, 3, 4)},
            "razao_previsao_media_movel_4": raw / mm4 if mm4 else float("nan"),
            "acima_maximo_historico": int(high),
            "modelo_menor_mae_na_serie": leader["modelo"],
            "mae_melhor_na_serie": float(leader["mae"]),
            "mae_operacional_na_serie": float(selected.iloc[0]["mae"]),
            "status_auditoria": "REVISAR" if reasons else "SEM_ALERTA_PELAS_REGRAS",
            "motivos": "; ".join(reasons),
            "modelo_ou_quantidade_alterados": "não",
        })
    history["data_corte"] = cutoff
    _require(source_data, ["data_venda", "feira", "categoria", "produto",
                            "quantidade_vendida", "quantidade_produzida",
                            "quantidade_sobra"], "Registros MySQL")
    product_history = source_data.copy()
    product_history["data_venda"] = pd.to_datetime(product_history["data_venda"], errors="raise")
    if product_history["data_venda"].isna().any():
        raise ValueError("Registros MySQL com datas ausentes.")
    product_history = product_history.loc[product_history["data_venda"].le(cutoff)].copy()
    product_columns = [c for c in ["id_operacao", "data_producao", "data_venda",
                       "feira", "categoria", "id_produto", "produto",
                       "quantidade_produzida", "quantidade_vendida", "quantidade_sobra",
                       "clima", "eh_feriado", "nome_feriado"] if c in product_history]
    product_history = product_history[product_columns].sort_values(["feira", "categoria", "produto", "data_venda"])
    product_history["data_corte"] = cutoff
    return {
        "auditoria_previsao_por_serie": pd.DataFrame(rows),
        "comparacao_modelos_por_serie": metrics,
        "auditoria_atributos_futuros": attributes,
        "historico_atual_categoria_feira": history,
        "historico_atual_produto_feira": product_history,
    }


def save_forecast_audit_outputs(
    analyses: dict[str, pd.DataFrame],
    output_directory: str | Path | None = None,
) -> list[Path]:
    destination = (Path(output_directory) if output_directory is not None else
                   Path(__file__).resolve().parent.parent / "reports" / "audit")
    destination.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, frame in analyses.items():
        path = destination / f"{name}.csv"
        frame.to_csv(path, index=False, encoding="utf-8-sig")
        paths.append(path)
    return paths
