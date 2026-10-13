"""Testes sintéticos da auditoria, sem conexão ao MySQL."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from pandas.testing import assert_frame_equal

from forecasting.operational_forecast import (
    create_future_feature_rows, create_future_schedule, prepare_operational_history,
    run_operational_forecast, save_operational_forecast_outputs,
)
from forecasting.product_distribution import save_product_distribution_outputs
from reporting.forecast_audit import (
    compare_models_by_series, run_forecast_audit, save_forecast_audit_outputs,
)
from tests.test_operational_forecast_dynamic import (
    _build_category_fair_data, _build_future_context, _build_phase10_decision,
)


def _case():
    base = _build_category_fair_data()
    base["total_produzido"] = base["demanda_observada"] + 3
    base["total_sobra"] = 3
    operational = run_operational_forecast(
        base, _build_phase10_decision("media_movel_3", "Média móvel 3"), 216,
    )
    forecasts = operational["previsoes_operacionais_categoria_feira"]
    cutoff = pd.Timestamp(forecasts["data_corte"].iloc[0])
    rows = []
    for index, forecast in forecasts.iterrows():
        for model in ("media_movel_3", "mlp_global"):
            for fold in (1, 2):
                rows.append({
                    "modelo": model, "nome_modelo": model,
                    "serie": forecast["serie"], "feira": forecast["feira"],
                    "categoria": forecast["categoria"], "fold": fold,
                    "id_linha_modelagem": index * 2 + fold,
                    "data_venda": cutoff - pd.Timedelta(days=2 - fold),
                    "valor_real": 10.0,
                    "previsao": 10.0 if model == "media_movel_3" else 30.0,
                })
    source = base[["data_venda", "data_producao", "feira", "categoria"]].copy()
    source["produto"] = "Produto sintético"
    source["quantidade_vendida"] = base["demanda_observada"]
    source["quantidade_produzida"] = base["total_produzido"]
    source["quantidade_sobra"] = 3
    return base, operational, pd.DataFrame(rows), source


class ForecastAuditTests(unittest.TestCase):
    def test_extreme_forecast_flagged_without_changing_model_or_values(self):
        base, operational, comparison, source = _case()
        forecasts = operational["previsoes_operacionais_categoria_feira"]
        mask = forecasts["serie"].eq("DOM_C | Doce")
        forecasts.loc[mask, ["previsao_bruta", "previsao_operacional"]] = 640
        original = forecasts.copy(deep=True)
        result = run_forecast_audit(base, operational, comparison, source)
        assert_frame_equal(forecasts, original)
        row = result["auditoria_previsao_por_serie"].set_index("serie").loc["DOM_C | Doce"]
        self.assertEqual(row["status_auditoria"], "REVISAR")
        self.assertEqual(row["previsao_operacional"], 640)
        self.assertEqual(row["acima_maximo_historico"], 1)
        self.assertEqual(row["modelo_operacional"], "media_movel_3")
        self.assertEqual(row["modelo_menor_mae_na_serie"], "media_movel_3")
        self.assertAlmostEqual(row["media_produzida_historica"], row["media_vendida_historica"] + 3)

    def test_future_rows_excluded_from_history_and_baselines(self):
        base, operational, comparison, source = _case()
        expected = run_forecast_audit(base, operational, comparison, source)
        extra = base.iloc[[0]].copy()
        extra["data_venda"] = base["data_venda"].max() + pd.Timedelta(days=30)
        extra["demanda_observada"] = 999999
        extra_source = source.iloc[[0]].copy()
        extra_source["data_venda"] = extra["data_venda"].iloc[0]
        result = run_forecast_audit(pd.concat([base, extra]), operational, comparison,
                                    pd.concat([source, extra_source]))
        assert_frame_equal(result["auditoria_previsao_por_serie"], expected["auditoria_previsao_por_serie"])
        self.assertEqual(len(result["historico_atual_produto_feira"]), len(source))

    def test_zero_targets_and_history_do_not_create_infinite_ratios(self):
        base, operational, comparison, source = _case()
        series = "DOM_E | Doce"
        base.loc[base["serie"].eq(series), "demanda_observada"] = 0
        comparison.loc[comparison["serie"].eq(series), "valor_real"] = 0
        result = run_forecast_audit(base, operational, comparison, source)
        row = result["auditoria_previsao_por_serie"].set_index("serie").loc[series]
        self.assertTrue(pd.isna(row["razao_previsao_media_movel_4"]))
        metrics = result["comparacao_modelos_por_serie"]
        metrics = metrics.loc[metrics["serie"].eq(series)]
        self.assertTrue(metrics["mape"].isna().all())
        self.assertTrue(metrics["wape"].isna().all())
        self.assertTrue(metrics["quantidade_alvos_zero"].eq(2).all())

    def test_different_cases_or_targets_rejected(self):
        _, operational, comparison, _ = _case()
        cutoff = operational["previsoes_operacionais_categoria_feira"]["data_corte"].iloc[0]
        with self.assertRaisesRegex(ValueError, "mesmos casos"):
            compare_models_by_series(comparison.iloc[1:], cutoff)
        changed = comparison.copy()
        changed.loc[0, "valor_real"] = 11
        with self.assertRaisesRegex(ValueError, "divergem"):
            compare_models_by_series(changed, cutoff)

    def test_mlp_features_audited_and_bad_lag_rejected(self):
        base, operational, comparison, source = _case()
        history, cutoff = prepare_operational_history(base, 4)
        future = create_future_feature_rows(history, create_future_schedule(history, cutoff),
                                            _build_future_context())
        operational["atributos_futuros_previsao"] = future
        operational["previsoes_operacionais_categoria_feira"]["modelo"] = "mlp_global"
        result = run_forecast_audit(base, operational, comparison, source)
        attributes = result["auditoria_atributos_futuros"]
        lags = attributes.loc[attributes["atributo"].eq("lag_1")]
        self.assertEqual(len(lags), 18)
        self.assertTrue(lags["verificacao"].eq("OK").all())
        trend = attributes.loc[attributes["atributo"].eq("tendencia_serie")]
        self.assertTrue(trend["observacao"].str.contains("esperado").all())
        future.loc[0, "lag_1"] += 1
        with self.assertRaisesRegex(ValueError, "lag_1 inconsistente"):
            run_forecast_audit(base, operational, comparison, source)

    def test_reports_saved_and_empty_attribute_report_has_headers(self):
        base, operational, comparison, source = _case()
        analyses = run_forecast_audit(base, operational, comparison, source)
        with tempfile.TemporaryDirectory() as directory:
            paths = save_forecast_audit_outputs(analyses, directory)
            self.assertEqual(len(paths), 5)
            self.assertTrue(all(path.exists() for path in paths))
            attributes = pd.read_csv(Path(directory) / "auditoria_atributos_futuros.csv")
            self.assertTrue(attributes.empty)
            self.assertIn("atributo", attributes)

    def test_climate_encoding_difference_is_visible_without_normalizing_inputs(self):
        base, operational, comparison, source = _case()
        history, cutoff = prepare_operational_history(base, 4)
        future = create_future_feature_rows(history, create_future_schedule(history, cutoff),
                                            _build_future_context())
        # Simula a regressão antiga após a preparação. A auditoria deve
        # registrar a entrada incorreta sem corrigi-la silenciosamente.
        future["clima"] = "Sol"
        operational["atributos_futuros_previsao"] = future
        operational["previsoes_operacionais_categoria_feira"]["modelo"] = "mlp_global"
        before = future.copy(deep=True)
        result = run_forecast_audit(base, operational, comparison, source)
        assert_frame_equal(before, future)
        climate = result["auditoria_atributos_futuros"]
        climate = climate.loc[climate["atributo"].eq("clima")]
        self.assertEqual(len(climate), 18)
        self.assertTrue(climate["fora_faixa_treino"].eq(1).all())
        self.assertTrue(climate["observacao"].str.contains("Categoria não vista").all())

    def test_savers_overwrite_stale_forecasts_in_same_project(self):
        _, operational, _, _ = _case()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch("forecasting.operational_forecast.__file__", str(root / "forecasting/operational_forecast.py")):
                save_operational_forecast_outputs(operational)
                current = operational["previsoes_operacionais_categoria_feira"].copy()
                current["modelo"] = "mlp_global"
                save_operational_forecast_outputs({**operational, "previsoes_operacionais_categoria_feira": current})
            saved = pd.read_csv(root / "reports/forecasts/previsoes_operacionais_categoria_feira.csv")
            self.assertTrue(saved["modelo"].eq("mlp_global").all())
            attributes = pd.read_csv(root / "reports/forecasts/atributos_futuros_previsao.csv")
            self.assertTrue(attributes.empty)
            with patch("forecasting.product_distribution.__file__", str(root / "forecasting/product_distribution.py")):
                for cut in ["2026-09-20", "2026-10-04"]:
                    save_product_distribution_outputs({"proporcoes_historicas_produto_feira": pd.DataFrame({"data_corte": [cut]})})
            proportions = pd.read_csv(root / "reports/forecasts/proporcoes_historicas_produto_feira.csv")
            self.assertEqual(proportions["data_corte"].iloc[0], "2026-10-04")


if __name__ == "__main__":
    unittest.main()
