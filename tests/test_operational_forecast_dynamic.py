"""Testes da previsão operacional dinâmica após a atualização do ranking."""

import unittest

import pandas as pd

from forecasting.operational_forecast import (
    CLIMATE_OPTIONS,
    FAIR_ORDER,
    collect_future_context,
    create_future_schedule,
    prepare_future_context,
    prepare_operational_history,
    run_operational_forecast,
)
from preprocessing.features import (
    add_historical_features,
    add_temporal_features,
)


CATEGORIES = ["Comum", "Doce", "Especial"]
FAIR_OFFSETS = {
    "QUA": 2,
    "QUI": 3,
    "SAB_C": 5,
    "SAB_E": 5,
    "DOM_C": 6,
    "DOM_E": 6,
}


def _build_category_fair_data(weeks: int = 12) -> pd.DataFrame:
    rows = []
    monday = pd.Timestamp("2026-01-05")

    for fair_index, fair in enumerate(FAIR_ORDER):
        for category_index, category in enumerate(CATEGORIES):
            series = f"{fair} | {category}"
            for week in range(weeks):
                sale_date = monday + pd.Timedelta(
                    days=FAIR_OFFSETS[fair] + 7 * week
                )
                demand = (
                    30
                    + fair_index * 12
                    + category_index * 8
                    + week * 3
                    + ((week + fair_index + category_index) % 3) * 2
                )
                rows.append(
                    {
                        "data_venda": sale_date,
                        "data_producao": sale_date - pd.Timedelta(days=1),
                        "serie": series,
                        "feira": fair,
                        "categoria": category,
                        "clima": CLIMATE_OPTIONS[
                            (week + fair_index) % len(CLIMATE_OPTIONS)
                        ],
                        "eh_feriado": int(week == 5),
                        "nome_feriado": "Teste" if week == 5 else "",
                        "demanda_observada": float(demand),
                    }
                )

    base = pd.DataFrame(rows).sort_values(
        ["serie", "data_venda"]
    ).reset_index(drop=True)
    return add_historical_features(add_temporal_features(base))


def _build_phase10_decision(model_key: str, model_name: str) -> dict:
    return {
        "decisao_modelo_fase10": pd.DataFrame(
            [
                {
                    "modelo_lider_fase10": model_key,
                    "nome_modelo_lider": model_name,
                    "mae_lider": 10.0,
                    "mse_lider": 150.0,
                    "rmse_lider": 12.2474,
                    "mape_lider": 9.5,
                }
            ]
        )
    }


def _build_future_context() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "feira": fair,
                "clima": CLIMATE_OPTIONS[index % len(CLIMATE_OPTIONS)],
                "eh_feriado": int(fair == "QUI"),
                "nome_feriado": "Feriado de teste" if fair == "QUI" else "",
            }
            for index, fair in enumerate(FAIR_ORDER)
        ]
    )


class DynamicOperationalForecastTests(unittest.TestCase):
    def test_mlp_operational_forecast_uses_explicit_future_context(self) -> None:
        data = _build_category_fair_data()
        analyses = run_operational_forecast(
            category_fair_data=data,
            phase10_analyses=_build_phase10_decision(
                "mlp_global", "MLP global"
            ),
            source_record_count=2000,
            future_context=_build_future_context(),
        )

        predictions = analyses[
            "previsoes_operacionais_categoria_feira"
        ]
        validation = analyses["validacao_previsao_operacional"]
        selection = analyses["selecao_mlp_operacional"]
        diagnostics = analyses["diagnostico_modelo_operacional"]

        self.assertEqual(len(predictions), 18)
        self.assertTrue(predictions["modelo"].eq("mlp_global").all())
        self.assertTrue(predictions["previsao_operacional"].ge(0).all())
        self.assertTrue(validation["resultado"].eq("OK").all())
        self.assertEqual(len(selection), 4)
        self.assertEqual(int(selection["selecionada"].sum()), 1)
        self.assertGreater(int(diagnostics["linhas_treino_modelo"].iloc[0]), 0)
        self.assertEqual(
            set(predictions["clima"]),
            set(_build_future_context()["clima"]),
        )

    def test_all_previous_models_have_operational_strategy(self) -> None:
        data = _build_category_fair_data()
        models = {
            "naive": "Naive (última ocorrência)",
            "media_movel_2": "Média móvel (2 ocorrências)",
            "media_movel_3": "Média móvel (3 ocorrências)",
            "media_movel_4": "Média móvel (4 ocorrências)",
            "suavizacao_exponencial_simples": (
                "Suavização exponencial simples"
            ),
            "regressao_linear_global": "Regressão linear global",
        }

        for model_key, model_name in models.items():
            with self.subTest(model=model_key):
                analyses = run_operational_forecast(
                    category_fair_data=data,
                    phase10_analyses=_build_phase10_decision(
                        model_key, model_name
                    ),
                    source_record_count=2000,
                    future_context=(
                        _build_future_context()
                        if model_key == "regressao_linear_global"
                        else None
                    ),
                )
                validation = analyses[
                    "validacao_previsao_operacional"
                ]
                self.assertTrue(validation["resultado"].eq("OK").all())

    def test_terminal_context_is_validated_and_normalized(self) -> None:
        data = _build_category_fair_data()
        history, cutoff = prepare_operational_history(data, 4)
        schedule = create_future_schedule(history, cutoff)
        answers = []

        for fair in FAIR_ORDER:
            answers.extend(
                [
                    "chuva moderada" if fair == "QUA" else "sol",
                    "n",
                ]
            )

        iterator = iter(answers)
        context = collect_future_context(
            schedule,
            input_function=lambda _prompt: next(iterator),
            output_function=lambda _message: None,
        )

        self.assertEqual(len(context), 6)
        self.assertEqual(context.loc[0, "clima"], "Chuva Moderado")
        self.assertTrue(context["eh_feriado"].eq(0).all())
        prepared = prepare_future_context(schedule, context)
        self.assertTrue(
            prepared["data_venda_prevista"].equals(
                schedule["data_venda_prevista"]
            )
        )

    def test_future_context_rejects_missing_holiday_name(self) -> None:
        data = _build_category_fair_data()
        history, cutoff = prepare_operational_history(data, 4)
        schedule = create_future_schedule(history, cutoff)
        context = _build_future_context()
        context.loc[context["feira"] == "QUI", "nome_feriado"] = ""

        with self.assertRaisesRegex(ValueError, "nome do feriado"):
            prepare_future_context(schedule, context)


if __name__ == "__main__":
    unittest.main()
