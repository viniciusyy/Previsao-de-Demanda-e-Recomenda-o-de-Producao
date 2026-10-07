"""Testes da consolidação operacional e documentação final ."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from optimization.production_recommendation import (
    run_production_recommendation,
)
from reporting.final_consolidation import (
    run_final_consolidation,
    save_final_consolidation_outputs,
)
from tests.test_production_recommendation import (
    _build_forecasts,
    _build_history,
)


def _build_ranking() -> pd.DataFrame:
    models = [
        ("mlp_global", "MLP global", 36.3347, 79.5986, 29.5914),
        (
            "media_movel_4",
            "Média móvel (4 ocorrências)",
            38.6250,
            74.0871,
            16.8128,
        ),
        (
            "suavizacao_exponencial_simples",
            "Suavização exponencial simples",
            39.5394,
            75.9570,
            16.8089,
        ),
        (
            "media_movel_3",
            "Média móvel (3 ocorrências)",
            41.5185,
            77.4082,
            17.4059,
        ),
        (
            "media_movel_2",
            "Média móvel (2 ocorrências)",
            41.7833,
            79.8557,
            17.3055,
        ),
        (
            "naive",
            "Naive (última ocorrência)",
            44.2333,
            89.5118,
            19.9933,
        ),
        (
            "regressao_linear_global",
            "Regressão linear global",
            46.3618,
            67.2343,
            82.2255,
        ),
    ]
    return pd.DataFrame(
        [
            {
                "posicao": position,
                "modelo": model,
                "nome_modelo": name,
                "mae": mae,
                "mse": rmse**2,
                "rmse": rmse,
                "mape": mape,
            }
            for position, (model, name, mae, rmse, mape) in enumerate(
                models, start=1
            )
        ]
    )


def _build_phase_inputs() -> tuple[
    pd.DataFrame,
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
]:
    history = _build_history()
    forecasts = _build_forecasts()
    forecasts["modelo_previsao_categoria"] = "mlp_global"
    forecasts["nome_modelo_previsao_categoria"] = "MLP global"

    recommendation_analyses = run_production_recommendation(
        source_data=history,
        product_forecasts=forecasts,
        source_record_count=len(history),
    )

    category_forecasts = (
        forecasts.groupby(
            [
                "data_corte",
                "data_producao_prevista",
                "data_venda_prevista",
                "feira",
                "categoria",
            ],
            as_index=False,
        )["previsao_produto"]
        .sum()
        .rename(columns={"previsao_produto": "previsao_operacional"})
    )
    category_forecasts["modelo"] = "mlp_global"
    category_forecasts["nome_modelo"] = "MLP global"

    future_context = (
        forecasts[["feira", "data_venda_prevista"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    future_context["clima"] = "Sol"
    future_context["eh_feriado"] = 0
    future_context["nome_feriado"] = ""

    operational_analyses = {
        "previsoes_operacionais_categoria_feira": category_forecasts,
        "validacao_previsao_operacional": pd.DataFrame(
            [
                {
                    "teste": f"Validação operacional {index}",
                    "resultado": "OK",
                    "detalhes": "Teste sintético aprovado.",
                }
                for index in range(1, 19)
            ]
        ),
        "decisao_modelo_operacional": pd.DataFrame(
            [
                {
                    "modelo_operacional": "mlp_global",
                    "nome_modelo_operacional": "MLP global",
                    "criterio_selecao": "menor MAE no walk-forward",
                    "mae_validacao": 36.3347,
                    "mse_validacao": 6335.9271,
                    "rmse_validacao": 79.5986,
                    "mape_validacao": 29.5914,
                    "registros_origem_mysql": len(history),
                    "data_corte": forecasts["data_corte"].max(),
                    "quantidade_series_previstas": 18,
                    "estrategia_operacional": "mlp_global",
                    "janela_historica": 4,
                    "linhas_treino_modelo_final": 270,
                    "configuracao_operacional_selecionada": (
                        "mlp_16_alpha_001"
                    ),
                    "contexto_futuro_requerido": "sim",
                }
            ]
        ),
        "contexto_futuro_previsao": future_context,
    }
    distribution_analyses = {
        "validacao_distribuicao_produtos": pd.DataFrame(
            [
                {
                    "teste": "Distribuição sintética",
                    "resultado": "OK",
                    "detalhes": "Totais reconciliados.",
                }
            ]
        )
    }

    return (
        _build_ranking(),
        operational_analyses,
        distribution_analyses,
        recommendation_analyses,
    )


class FinalConsolidationTests(unittest.TestCase):
    def test_final_package_is_reconciled_and_saved(self) -> None:
        (
            ranking,
            operational,
            distribution,
            recommendation,
        ) = _build_phase_inputs()

        artifacts = run_final_consolidation(
            ranking=ranking,
            operational_analyses=operational,
            distribution_analyses=distribution,
            recommendation_analyses=recommendation,
            censored_count=700,
        )

        validation = artifacts["validacao_consolidacao_final"]
        summary = artifacts["resumo_execucao_final"].iloc[0]
        plan = artifacts["plano_producao_operacional"]
        report = artifacts["relatorio_operacional_final"]

        self.assertEqual(len(validation), 12)
        self.assertTrue(validation["resultado"].eq("OK").all())
        self.assertEqual(
            int(plan["recomendacao_producao"].sum()),
            int(summary["recomendacao_total"]),
        )
        self.assertTrue(plan["produto_permitido"].eq(1).all())
        self.assertIn("MLP global", report)
        self.assertIn("Capacidades utilizadas", report)
        self.assertIn("Premissas, decisões e limitações", report)

        with tempfile.TemporaryDirectory() as temporary_directory:
            generated = save_final_consolidation_outputs(
                artifacts,
                output_directory=temporary_directory,
            )
            self.assertEqual(len(generated), 5)
            self.assertTrue(all(path.exists() for path in generated))
            report_path = Path(temporary_directory) / (
                "relatorio_operacional_final.md"
            )
            self.assertIn(
                "Relatório operacional final",
                report_path.read_text(encoding="utf-8"),
            )

    def test_leader_mismatch_interrupts_consolidation(self) -> None:
        (
            ranking,
            operational,
            distribution,
            recommendation,
        ) = _build_phase_inputs()
        ranking.loc[ranking["posicao"].eq(1), "modelo"] = "media_movel_4"

        with self.assertRaisesRegex(ValueError, "Modelo operacional"):
            run_final_consolidation(
                ranking=ranking,
                operational_analyses=operational,
                distribution_analyses=distribution,
                recommendation_analyses=recommendation,
                censored_count=700,
            )

    def test_upstream_failure_interrupts_consolidation(self) -> None:
        (
            ranking,
            operational,
            distribution,
            recommendation,
        ) = _build_phase_inputs()
        distribution["validacao_distribuicao_produtos"].loc[
            0, "resultado"
        ] = "FALHA"

        with self.assertRaisesRegex(ValueError, "Validações anteriores"):
            run_final_consolidation(
                ranking=ranking,
                operational_analyses=operational,
                distribution_analyses=distribution,
                recommendation_analyses=recommendation,
                censored_count=700,
            )


if __name__ == "__main__":
    unittest.main()

