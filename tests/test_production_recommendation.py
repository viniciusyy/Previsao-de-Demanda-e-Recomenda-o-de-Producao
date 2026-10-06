"""Testes da recomendação de produção ."""

import unittest

import pandas as pd

from optimization.production_recommendation import (
    CAPACITY_SOURCE_MANUAL,
    run_production_recommendation,
)


FAIRS = ["QUA", "QUI", "SAB_C", "SAB_E", "DOM_C", "DOM_E"]
RESTRICTED = {
    ("SAB_E", 5),
    ("SAB_E", 15),
    ("DOM_E", 5),
    ("DOM_E", 15),
}


def _category(product_id: int) -> str:
    if product_id <= 19:
        return "Comum"
    if product_id <= 24:
        return "Especial"
    return "Doce"


def _build_forecasts() -> pd.DataFrame:
    cutoff = pd.Timestamp("2026-09-20")
    sale_dates = {
        "QUA": pd.Timestamp("2026-09-23"),
        "QUI": pd.Timestamp("2026-09-24"),
        "SAB_C": pd.Timestamp("2026-09-26"),
        "SAB_E": pd.Timestamp("2026-09-26"),
        "DOM_C": pd.Timestamp("2026-09-27"),
        "DOM_E": pd.Timestamp("2026-09-27"),
    }
    rows = []

    for fair in FAIRS:
        for product_id in range(1, 28):
            allowed = (fair, product_id) not in RESTRICTED
            rows.append(
                {
                    "data_corte": cutoff,
                    "data_producao_prevista": (
                        sale_dates[fair] - pd.Timedelta(days=1)
                    ),
                    "data_venda_prevista": sale_dates[fair],
                    "feira": fair,
                    "categoria": _category(product_id),
                    "id_produto": product_id,
                    "produto": f"Produto {product_id}",
                    "produto_permitido": int(allowed),
                    "motivo_restricao": (
                        "" if allowed else "Restrição comercial"
                    ),
                    "previsao_produto": (
                        6 + product_id % 7 if allowed else 0
                    ),
                    "modelo_previsao_categoria": "media_movel_4",
                    "nome_modelo_previsao_categoria": (
                        "Média móvel (4 ocorrências)"
                    ),
                    "participacao_percentual": 0.0,
                }
            )

    return pd.DataFrame(rows)


def _build_history() -> pd.DataFrame:
    rows = []
    operation_id = 1
    first_tuesday = pd.Timestamp("2026-08-04")

    for week in range(6):
        dates = {
            "QUA": first_tuesday + pd.Timedelta(days=7 * week),
            "QUI": first_tuesday + pd.Timedelta(days=1 + 7 * week),
            "SAB_C": first_tuesday + pd.Timedelta(days=3 + 7 * week),
            "SAB_E": first_tuesday + pd.Timedelta(days=3 + 7 * week),
            "DOM_C": first_tuesday + pd.Timedelta(days=4 + 7 * week),
            "DOM_E": first_tuesday + pd.Timedelta(days=4 + 7 * week),
        }
        saturday_total = 320 + 16 * week
        sunday_total = 330 + 20 * week
        totals = {
            "QUA": 180 + 8 * week,
            "QUI": 250 + 10 * week,
            "SAB_C": saturday_total // 2,
            "SAB_E": saturday_total - saturday_total // 2,
            "DOM_C": sunday_total // 2,
            "DOM_E": sunday_total - sunday_total // 2,
        }

        for fair in FAIRS:
            allowed_ids = [
                product_id
                for product_id in range(1, 28)
                if (fair, product_id) not in RESTRICTED
            ]
            base, remainder = divmod(totals[fair], len(allowed_ids))
            quantities = {product_id: 0 for product_id in range(1, 28)}

            for position, product_id in enumerate(allowed_ids):
                quantities[product_id] = base + int(position < remainder)

            for product_id in range(1, 28):
                rows.append(
                    {
                        "id_operacao": operation_id,
                        "data_producao": dates[fair],
                        "data_venda": dates[fair] + pd.Timedelta(days=1),
                        "feira": fair,
                        "id_produto": product_id,
                        "quantidade_produzida": quantities[product_id],
                    }
                )
            operation_id += 1

    return pd.DataFrame(rows)


class ProductionRecommendationTests(unittest.TestCase):
    def test_historical_capacity_and_simplex_solution(self) -> None:
        history = _build_history()
        forecasts = _build_forecasts()

        analyses = run_production_recommendation(
            source_data=history,
            product_forecasts=forecasts,
            source_record_count=len(history),
        )

        capacities = analyses["capacidades_estimadas_producao"].set_index(
            "grupo_producao"
        )["capacidade_estimada"]
        self.assertEqual(
            capacities.to_dict(),
            {
                "TERCA_QUA": 220,
                "QUARTA_QUI": 300,
                "SEXTA_SABADOS": 400,
                "SABADO_DOMINGOS": 430,
            },
        )

        validation = analyses["validacao_recomendacao_producao"]
        self.assertEqual(len(validation), 20)
        self.assertTrue(validation["resultado"].eq("OK").all())

        summary = analyses["resumo_recomendacao_por_dia"]
        expected_total = summary[[
            "previsao_total",
            "capacidade_estimada",
        ]].min(axis=1).astype(int)
        self.assertTrue(
            summary["recomendacao_total"].equals(expected_total)
        )

        recommendations = analyses[
            "recomendacoes_producao_produto_feira"
        ]
        restricted = recommendations.loc[
            recommendations["produto_permitido"].eq(0)
        ]
        self.assertEqual(len(restricted), 4)
        self.assertTrue(
            restricted["recomendacao_producao"].eq(0).all()
        )

    def test_manual_capacity_can_replace_historical_source(self) -> None:
        history = _build_history()
        forecasts = _build_forecasts()
        manual = {
            "TERCA_QUA": 1000,
            "QUARTA_QUI": 1000,
            "SEXTA_SABADOS": 1000,
            "SABADO_DOMINGOS": 1000,
        }

        analyses = run_production_recommendation(
            source_data=history,
            product_forecasts=forecasts,
            source_record_count=len(history),
            capacity_source=CAPACITY_SOURCE_MANUAL,
            manual_capacities=manual,
        )
        recommendations = analyses[
            "recomendacoes_producao_produto_feira"
        ]

        self.assertTrue(
            recommendations["recomendacao_producao"].equals(
                recommendations["previsao_produto"]
            )
        )
        self.assertTrue(
            recommendations["fonte_capacidade"].eq("manual").all()
        )

    def test_manual_capacity_requires_all_groups(self) -> None:
        with self.assertRaisesRegex(
            ValueError, "Capacidades manuais incompletas"
        ):
            run_production_recommendation(
                source_data=_build_history(),
                product_forecasts=_build_forecasts(),
                source_record_count=1,
                capacity_source=CAPACITY_SOURCE_MANUAL,
                manual_capacities={"TERCA_QUA": 1000},
            )


if __name__ == "__main__":
    unittest.main()
