"""Integração do fluxo principal com MySQL simulado e modelos reais."""

import contextlib
import io
import tempfile
import sys
import types
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

import forecasting.mlp_model
import forecasting.product_distribution
import optimization.production_recommendation
import reporting.final_consolidation
import reporting.forecast_audit

from forecasting.operational_forecast import run_operational_forecast
from tests.test_operational_forecast_dynamic import _build_category_fair_data, _build_future_context


class MainForecastAuditTests(unittest.TestCase):
    def test_main_updates_forecasts_and_writes_current_audit(self):
        base = _build_category_fair_data(weeks=15)
        rows = []
        for operation, (_, row) in enumerate(
            base.drop_duplicates(["data_venda", "feira"]).iterrows(), start=1
        ):
            for product in range(1, 28):
                if row["feira"].endswith("_E") and product in (5, 15):
                    continue
                sold = 3 + product % 4 + operation % 3
                rows.append({
                    "id_operacao": operation, "id_produto": product,
                    "produto": f"Produto {product}", "feira": row["feira"],
                    "categoria": "Comum" if product < 20 else "Especial" if product < 25 else "Doce",
                    "data_venda": row["data_venda"], "data_producao": row["data_producao"],
                    "quantidade_vendida": sold, "quantidade_produzida": sold + 1,
                    "quantidade_sobra": 1, "clima": {
                        "Sol": "SOL", "Frio": "FRIO", "Garoa": "GAROA",
                        "Chuva Moderado": "CHUVA_MODERADA", "Chuva Forte": "CHUVA_FORTE",
                    }[row["clima"]],
                    "eh_feriado": row["eh_feriado"], "nome_feriado": row["nome_feriado"],
                    "observacoes": "Base sintética de teste",
                })
        source = pd.DataFrame(rows)

        # O teste de integração não precisa do driver nem de um MySQL real.
        connection = types.ModuleType("database.connection")
        connection.test_connection = lambda: True
        repository = types.ModuleType("database.repository")
        repository.load_historical_data = lambda: source
        repository.load_referential_integrity_issues = lambda: pd.DataFrame()
        with patch.dict(sys.modules, {"database.connection": connection,
                                      "database.repository": repository}):
            import main as application

        def forecast_with_context(**kwargs):
            return run_operational_forecast(**kwargs, future_context=_build_future_context())

        with tempfile.TemporaryDirectory() as directory, contextlib.ExitStack() as stack:
            root = Path(directory)
            for module, relative in [
                ("forecasting.mlp_model", "forecasting/mlp_model.py"),
                ("forecasting.operational_forecast", "forecasting/operational_forecast.py"),
                ("forecasting.product_distribution", "forecasting/product_distribution.py"),
                ("optimization.production_recommendation", "optimization/production_recommendation.py"),
                ("reporting.forecast_audit", "reporting/forecast_audit.py"),
                ("reporting.final_consolidation", "reporting/final_consolidation.py"),
            ]:
                stack.enter_context(patch(f"{module}.__file__", str(root / relative)))
            stack.enter_context(patch.object(application, "test_connection", return_value=True))
            stack.enter_context(patch.object(application, "load_historical_data", return_value=source))
            stack.enter_context(patch.object(application, "load_referential_integrity_issues", return_value=pd.DataFrame()))
            stack.enter_context(patch.object(application, "run_operational_forecast", side_effect=forecast_with_context))
            stack.enter_context(patch.object(application, "generate_production_recommendation_plots", return_value=[]))
            console = io.StringIO()
            with contextlib.redirect_stdout(console):
                application.main()
            self.assertNotIn("Falha na execução", console.getvalue())
            category = pd.read_csv(root / "reports/forecasts/previsoes_operacionais_categoria_feira.csv")
            product = pd.read_csv(root / "reports/forecasts/previsoes_operacionais_produto_feira.csv")
            summary = pd.read_csv(root / "reports/final/resumo_execucao_final.csv")
            audit = pd.read_csv(root / "reports/audit/auditoria_previsao_por_serie.csv")
            cutoff = summary["data_corte"].iloc[0]
            for frame in (category, product, audit):
                self.assertTrue(frame["data_corte"].eq(cutoff).all())
            self.assertEqual(int(summary["registros_origem_mysql"].iloc[0]), len(source))
            self.assertEqual(len(audit), 18)
            self.assertTrue(category["modelo"].eq(summary["modelo_operacional"].iloc[0]).all())
            self.assertTrue((root / "reports/audit/historico_atual_produto_feira.csv").exists())


if __name__ == "__main__":
    unittest.main()
